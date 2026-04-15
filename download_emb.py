"""Download STaRK-Prime embeddings from Google Drive."""
import os
import gdown

query_id = '1MshwJttPZsHEM2cKA5T13SIrsLeBEdyU'
node_id  = '16EJvCMbgkVrQ0BuIBvLBp-BYPaye-Edy'

os.makedirs('emb/prime/text-embedding-ada-002/query', exist_ok=True)
os.makedirs('emb/prime/text-embedding-ada-002/doc',   exist_ok=True)

print("Downloading query embeddings (69MB)...")
gdown.download(
    f'https://drive.google.com/uc?id={query_id}',
    'emb/prime/text-embedding-ada-002/query/query_emb_dict.pt'
)

print("Downloading node embeddings (794MB)...")
gdown.download(
    f'https://drive.google.com/uc?id={node_id}',
    'emb/prime/text-embedding-ada-002/doc/candidate_emb_dict.pt'
)

print("Done.")
