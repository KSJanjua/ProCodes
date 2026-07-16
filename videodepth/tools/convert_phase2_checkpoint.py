"""Diagnose / repair a Phase-2 checkpoint saved under a different transformers
version.

Symptom (SMOKE=1 run_pipeline.sh step 4):
    RuntimeError: Error(s) in loading state_dict for Phase2Model:
        Missing key(s): ...attention.q_proj.weight ...
When phase2_run/best.pth was trained, HF Mask2Former's Swin attention used
different key names than the currently-installed transformers. The weights
are fine; only the KEYS moved. This tool aligns them.

Two modes:
  * report (default): print exactly which keys mismatch, categorised, plus a
    proposed remap derived by matching each still-unmatched checkpoint tensor
    to a model tensor **under the same parent module, in definition order,
    with identical shape** — unambiguous for the Swin attention rename.
  * --write OUT.pth: apply that remap and save a repaired checkpoint (verified
    to load with strict=True before writing). Never overwrites the input.

Usage:
    python -m videodepth.tools.convert_phase2_checkpoint \\
        --config instancedepth/configs/phase2_mask2former.yaml \\
        --checkpoint runs/phase2_run/best.pth                 # report

    python -m videodepth.tools.convert_phase2_checkpoint \\
        --config instancedepth/configs/phase2_mask2former.yaml \\
        --checkpoint runs/phase2_run/best.pth \\
        --write runs/phase2_run/best_converted.pth            # repair
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

log = logging.getLogger("videodepth.tools.convert_phase2_checkpoint")


def _group(key: str) -> str:
    """Group key = the module TWO levels up (the nearest common ancestor of a
    renamed submodule's parameter). The transformers rename moves the
    submodule name (``self.query`` -> ``q_proj``), so the parameter's
    grandparent (e.g. ``...attention``) is what stays shared across the
    rename. Falls back to the immediate parent for shallow keys."""
    parts = key.rsplit(".", 2)
    return parts[0] if len(parts) == 3 else key.rsplit(".", 1)[0]


def build_remap(ckpt_sd: Dict[str, torch.Tensor],
                model_sd: Dict[str, torch.Tensor]
                ) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Return (remap ckpt_key->model_key, unresolved_model_keys,
    unresolved_ckpt_keys). Keys present verbatim in both are left as-is
    (identity, not in remap). Remaining keys are matched within a shared
    parent module, in order, by shape."""
    model_keys = set(model_sd)
    ckpt_keys = set(ckpt_sd)
    shared = model_keys & ckpt_keys

    miss_model = [k for k in model_sd if k not in ckpt_keys]      # model wants, ckpt lacks
    extra_ckpt = [k for k in ckpt_sd if k not in model_keys]      # ckpt has, model lacks

    # group both sides by parent module
    m_by_parent: Dict[str, List[str]] = defaultdict(list)
    c_by_parent: Dict[str, List[str]] = defaultdict(list)
    for k in miss_model:
        m_by_parent[_group(k)].append(k)
    for k in extra_ckpt:
        c_by_parent[_group(k)].append(k)

    remap: Dict[str, str] = {}
    used_ckpt: set = set()
    for group, m_keys in m_by_parent.items():
        c_keys = c_by_parent.get(group, [])
        # match in definition order, by shape (unambiguous for q/k/v etc.)
        for mk in m_keys:
            for ck in c_keys:
                if ck in used_ckpt:
                    continue
                if ckpt_sd[ck].shape == model_sd[mk].shape:
                    remap[ck] = mk
                    used_ckpt.add(ck)
                    break

    unresolved_model = [k for k in miss_model if k not in remap.values()]
    unresolved_ckpt = [k for k in extra_ckpt if k not in used_ckpt]
    log.info("keys: %d identical, %d remapped, %d model-unresolved, %d ckpt-unused",
             len(shared), len(remap), len(unresolved_model), len(unresolved_ckpt))
    return remap, unresolved_model, unresolved_ckpt


def apply_remap(ckpt_sd: Dict[str, torch.Tensor], remap: Dict[str, str],
                model_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """New state dict keyed by model names: identity keys kept, remapped keys
    renamed, ckpt-only keys dropped."""
    out = {}
    for k, v in ckpt_sd.items():
        if k in remap:
            out[remap[k]] = v
        elif k in model_sd:
            out[k] = v
        # else: ckpt-only (e.g. relative_position_index buffer) -> drop
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--write", default=None, help="output path for the repaired checkpoint")
    ap.add_argument("--samples", type=int, default=8, help="how many example pairs to print")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from instancedepth.configs.phase2_config import Phase2Config
    from instancedepth.models.phase2.model import Phase2Model

    cfg = Phase2Config.from_yaml(args.config)
    model = Phase2Model(
        checkpoint=cfg.model.checkpoint, checkpoint_dir=cfg.model.checkpoint_dir,
        allow_hub_download=cfg.model.allow_hub_download, num_classes=cfg.model.num_classes,
    )
    model_sd = model.state_dict()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_sd = ckpt["model"]

    remap, unresolved_model, unresolved_ckpt = build_remap(ckpt_sd, model_sd)

    print("\n--- example remaps (ckpt key -> model key) ---")
    for ck, mk in list(remap.items())[:args.samples]:
        print(f"  {ck}\n    -> {mk}   {tuple(ckpt_sd[ck].shape)}")
    if unresolved_model:
        print(f"\n--- {len(unresolved_model)} MODEL keys with no checkpoint match "
              f"(would be left at init) ---")
        for k in unresolved_model[:args.samples]:
            print(f"  {k}   {tuple(model_sd[k].shape)}")
    if unresolved_ckpt:
        print(f"\n--- {len(unresolved_ckpt)} checkpoint keys unused (dropped) ---")
        for k in unresolved_ckpt[:args.samples]:
            print(f"  {k}   {tuple(ckpt_sd[k].shape)}")

    # Only real (non-buffer) unresolved model keys are dangerous. Buffers HF
    # recomputes (relative_position_index) are safe to leave.
    dangerous = [k for k in unresolved_model
                 if not k.endswith("relative_position_index")]
    if dangerous:
        print(f"\n[!] {len(dangerous)} model WEIGHT keys have no source in the "
              f"checkpoint — a converted checkpoint would leave these at random "
              f"init. This is NOT a clean transformers-rename; investigate before "
              f"trusting a conversion. First few:\n    " +
              "\n    ".join(dangerous[:args.samples]))
    else:
        print("\n[OK] Every model weight is covered — this is a clean key-rename; "
              "conversion is safe.")

    if args.write:
        assert Path(args.write).resolve() != Path(args.checkpoint).resolve(), \
            "refusing to overwrite the input checkpoint"
        new_sd = apply_remap(ckpt_sd, remap, model_sd)
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        real_missing = [k for k in missing if not k.endswith("relative_position_index")]
        if real_missing:
            raise RuntimeError(
                f"converted checkpoint still misses {len(real_missing)} weight keys "
                f"(e.g. {real_missing[:3]}) — not writing. Run in report mode and "
                f"share the output.")
        out_ckpt = dict(ckpt)
        out_ckpt["model"] = new_sd
        torch.save(out_ckpt, args.write)
        log.info("wrote repaired checkpoint -> %s (loads with %d recomputed buffers, "
                 "%d dropped ckpt keys)", args.write, len(missing), len(unexpected))
        print(f"\nUse it:  --override phase2_checkpoint={args.write}")


if __name__ == "__main__":
    main()
