import os, requests, time, json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

FOLDER_ID = "16NQZrW1raRxJJNPknh0OVlyrl3JdeJy8"
API_KEY   = "AIzaSyDFciB1s214mapwUAI2NybDunC68GXfwBM"
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Saving to: {OUT_DIR}")

# Fetch all file listings
all_files, page_token = [], None
while True:
    params = {
        "q": f"'{FOLDER_ID}' in parents and trashed=false",
        "fields": "nextPageToken, files(id, name)",
        "pageSize": 1000,
        "key": API_KEY,
    }
    if page_token:
        params["pageToken"] = page_token
    data = requests.get("https://www.googleapis.com/drive/v3/files", params=params, timeout=30).json()
    if "error" in data:
        print("Error:", data["error"]["message"])
        break
    all_files.extend(data.get("files", []))
    page_token = data.get("nextPageToken")
    if not page_token:
        break

already = set(os.listdir(OUT_DIR))
todo = [f for f in all_files if f["name"].endswith(".pt") and f["name"] not in already]
print(f"Total: {len(all_files)} | Already have: {len(already)} | To download: {len(todo)}")

if not todo:
    print("Nothing to download!")
    exit(0)

def download_one(f):
    out_path = os.path.join(OUT_DIR, f["name"])
    if os.path.exists(out_path):
        return "skip"
    url = f"https://www.googleapis.com/drive/v3/files/{f['id']}?alt=media&key={API_KEY}"
    for attempt in range(1, 6):
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            return "ok"
        except Exception as e:
            if attempt == 5:
                return f"FAILED:{e}"
            time.sleep(2 ** attempt)

failed = []
with ThreadPoolExecutor(max_workers=3) as pool:
    futures = {pool.submit(download_one, f): f["name"] for f in todo}
    with tqdm(total=len(todo), unit="file") as bar:
        for fut in as_completed(futures):
            result = fut.result()
            bar.set_postfix_str(futures[fut][:40])
            bar.update(1)
            if result and "FAILED" in result:
                failed.append((futures[fut], result))

print(f"\nDone. Failed: {len(failed)}")
if failed:
    with open("failed_downloads.json", "w") as fh:
        json.dump(failed, fh, indent=2)
    print("Failed files saved to failed_downloads.json")

print(f"Total .pt files in {OUT_DIR}: {len([f for f in os.listdir(OUT_DIR) if f.endswith('.pt')])}")
