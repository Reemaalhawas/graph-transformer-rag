"""
Quick debug script to diagnose all-zeros metrics.
Run BEFORE train.py to see what the model actually generates.

Usage:
    python debug_eval.py --checkpoint results/gat_seed42/best_model.pt
    python debug_eval.py  # no checkpoint: just test raw Llama
"""

import os, sys, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_dir',   type=str, default='data/processed')
    parser.add_argument('--n',          type=int, default=5, help='samples to test')
    args = parser.parse_args()

    # ── 1. Load a few subgraphs ─────────────────────────────────────────────
    files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.pt')])[:args.n]
    samples = []
    for f in files:
        data = torch.load(os.path.join(args.data_dir, f), weights_only=False)
        label = getattr(data, 'label', None) or ''
        question = getattr(data, 'question', '')
        samples.append({'question': question, 'label': label, 'file': f})

    print("\n===== SAMPLE LABELS =====")
    for s in samples:
        print(f"File:     {s['file']}")
        print(f"Question: {s['question'][:100]}")
        print(f"Label:    {repr(s['label'])}")
        print()

    # ── 2. Load LLM ──────────────────────────────────────────────────────────
    llm_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {llm_name} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for inference

    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        # Only load LLM weights (GNN is separate)
        llm_state = {k.replace('llm.', ''): v for k, v in ckpt.items() if k.startswith('llm.')}
        if llm_state:
            llm.load_state_dict(llm_state, strict=False)
            print(f"Loaded {len(llm_state)} LLM keys from checkpoint")
        else:
            print("No LLM keys found in checkpoint — using raw LLM")

    device = next(iter(llm.parameters())).device

    # ── 3. Baseline inference (no graph) ─────────────────────────────────────
    print("\n===== BASELINE PREDICTIONS (no graph) =====")
    for s in samples:
        prompt = f"Question: {s['question']}\n\nAnswer:"
        enc = tokenizer(prompt, return_tensors='pt',
                        truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = llm.generate(
                **enc,
                max_new_tokens=64,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        # slice off input tokens
        new_tokens = out[:, enc['input_ids'].size(1):]
        pred = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        print(f"Q:     {s['question'][:80]}")
        print(f"Pred:  {repr(pred)}")
        print(f"Label: {repr(s['label'])}")
        print()

    # ── 4. Test inputs_embeds generation (mimics g_retriever mode) ───────────
    print("\n===== inputs_embeds GENERATION TEST =====")
    s = samples[0]
    prompt = f"Question: {s['question']}\n\nAnswer:"
    enc = tokenizer(prompt, return_tensors='pt',
                    truncation=True, max_length=256,
                    padding='max_length').to(device)
    input_ids = enc['input_ids']
    attn_mask  = enc['attention_mask']

    text_emb  = llm.get_input_embeddings()(input_ids)          # [1, L, H]
    # Prepend a dummy graph token (zeros)
    graph_emb = torch.zeros(1, 1, text_emb.size(-1),
                             device=device, dtype=text_emb.dtype)
    inputs_emb = torch.cat([graph_emb, text_emb], dim=1)       # [1, L+1, H]
    graph_mask = torch.ones(1, 1, device=device, dtype=attn_mask.dtype)
    attn_mask2 = torch.cat([graph_mask, attn_mask], dim=1)

    with torch.no_grad():
        out = llm.generate(
            inputs_embeds=inputs_emb,
            attention_mask=attn_mask2,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    print(f"inputs_embeds shape: {inputs_emb.shape}")
    print(f"generate() output shape: {out.shape}")
    print(f"Output[:10 tokens]: {out[0, :10].tolist()}")

    # Try both slicing strategies
    pred_full  = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    pred_slice = tokenizer.decode(out[0, inputs_emb.size(1):], skip_special_tokens=True).strip()

    print(f"\nDecode FULL output:         {repr(pred_full[:100])}")
    print(f"Decode SLICED [input_len:]: {repr(pred_slice[:100])}")
    print(f"Label:                      {repr(s['label'])}")

if __name__ == '__main__':
    main()
