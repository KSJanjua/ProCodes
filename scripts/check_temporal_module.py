"""How far did the temporal module actually move from its zero-init identity?

Zero-init means proj_out starts at exactly 0 -> the aligner is a no-op. If it
is still ~0 after training, the module never learned to do anything and a null
TAE result is fully explained (no bug, no metric issue -- just no gradient
pressure). Run from the repo root.
"""
import sys, torch

ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "runs/hdi_temporal/best.pth",
                map_location="cpu", weights_only=False)["model"]
tk = {k: v for k, v in ck.items() if "temporal_aligners" in k}
print(f"temporal tensors found: {len(tk)}")
if not tk:
    print("!! no temporal weights in this checkpoint")
    raise SystemExit
for k, v in tk.items():
    if "proj_out" in k:
        print(f"  {k:64s} absmax={v.abs().max():.6f}  absmean={v.abs().mean():.6f}")
out = torch.cat([v.flatten() for k, v in tk.items() if "proj_out" in k])
print(f"\nproj_out overall: absmax={out.abs().max():.6f}  absmean={out.abs().mean():.6f}")
print("VERDICT:", "still ~identity -> module learned ~nothing"
      if out.abs().max() < 1e-3 else "moved away from identity -> it did learn something")
