"""L3 SSL：冻结 ESM-2 t33 + bottleneck adapter MLM 预训练（8GB 默认 bs=8）。
P9：与 Bernett 蛋白精确哈希比对剔除重复。前置：prep/ssl_prepare.sh 完成 MMseqs2 过滤。"""
import argparse, hashlib, os, random as pyrandom
import torch, torch.nn as nn, torch.nn.functional as F
import esm
from model import SeqAdapter
from config import Paths, FeatureModels, save_json

def read_fasta(path, cap=4_000_000):
    seqs, name, buf = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    assert len(seqs) <= cap, "SSL corpus exceeds cap"
    return seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)          # 8GB 安全默认
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--max-batches-per-epoch", type=int, default=0)
    ap.add_argument("--bernett-fasta", default="")
    a = ap.parse_args()
    paths = Paths()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    bc = alphabet.get_batch_converter()

    data = [(k, v[:a.max_len]) for k, v in read_fasta(paths.ssl_fasta).items() if len(v) > 0]
    if a.bernett_fasta:
        ban = {hashlib.md5(s.upper().encode()).hexdigest() for s in read_fasta(a.bernett_fasta).values()}
        n0 = len(data)
        data = [(k, s) for k, s in data if hashlib.md5(s.upper().encode()).hexdigest() not in ban]
        print(f"P9 exact-dup filter: {n0} -> {len(data)} (removed {n0 - len(data)})")

    adapter = SeqAdapter(1280).to(dev)
    mlm_head = nn.Linear(1280, len(alphabet)).to(dev)
    opt = torch.optim.AdamW(list(adapter.parameters()) + list(mlm_head.parameters()),
                            lr=a.lr, weight_decay=1e-6)

    mask_id = alphabet.mask_idx
    special = {alphabet.padding_idx, alphabet.cls_idx, alphabet.eos_idx, alphabet.unk_idx, mask_id}
    special_t = torch.tensor(list(special), device=dev)
    valid_toks = [t for t in alphabet.standard_toks
                  if t not in (".", "-") and not t.startswith("<")]
    valid_ids = torch.tensor([alphabet.tok_to_idx[t] for t in valid_toks], device=dev)

    rng = pyrandom.Random(42)
    for ep in range(a.epochs):
        rng.shuffle(data)
        nb = tot = 0
        for k in range(0, len(data) - a.bs + 1, a.bs):
            chunk = data[k:k + a.bs]
            _, _, toks = bc(chunk)
            toks = toks.to(dev)
            labels = toks.clone()
            maskable = ~torch.isin(toks, special_t)
            m = maskable & (torch.rand_like(toks, dtype=torch.float) < 0.15)
            rnd = valid_ids[torch.randint(0, len(valid_ids), toks.shape, device=dev)]
            r = torch.rand_like(toks, dtype=torch.float)
            input_toks = toks.masked_fill(m, mask_id)                 # 80% mask
            input_toks = torch.where(m & (r < 0.1), rnd, input_toks)  # 10% random
            input_toks = torch.where(m & (r >= 0.9), toks, input_toks)  # 10% unchanged
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                out = model(input_toks, repr_layers=[model.num_layers])
            h = adapter(out["representations"][model.num_layers].float())
            logits = mlm_head(h)
            loss = F.cross_entropy(logits[m], labels[m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1; print(f"[{__import__('time').strftime('%H:%M:%S')}] ep{ep} nb={nb} avg={tot/nb:.4f}", flush=True) if nb % 500 == 0 else None
            if a.max_batches_per_epoch and nb >= a.max_batches_per_epoch:
                break
        print(f"SSL ep{ep}: loss={tot / max(nb, 1):.4f} batches={nb}")
    save_json({"status": "ok", "n_ssl_seqs": len(data)}, paths.adapter_ckpt + ".meta.json")
    torch.save(adapter.state_dict(), paths.adapter_ckpt)
    print("saved →", paths.adapter_ckpt)

if __name__ == "__main__":
    main()
