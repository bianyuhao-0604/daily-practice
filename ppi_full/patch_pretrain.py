s = open("ssl_pretrain.py", encoding="utf-8").read()
s2 = s.replace("cap=2_000_000", "cap=4_000_000")
assert s2 != s, "cap anchor missing"
old = "            nb += 1"
new = ("            nb += 1; print(f\"[{__import__('time').strftime('%H:%M:%S')}] "
       "ep{ep} nb={nb} avg={tot/nb:.4f}\", flush=True) if nb % 500 == 0 else None")
s3 = s2.replace(old, new)
assert s3 != s2, "progress anchor missing"
open("ssl_pretrain.py", "w", encoding="utf-8").write(s3)
print("patched OK: cap 2M->4M + progress every 500 batches")
