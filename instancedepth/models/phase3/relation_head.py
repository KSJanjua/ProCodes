"""Occlusion Pair Relation Reasoning head Phi_o (paper Eq. 8-9) + the
non-differentiable composite-back used at inference/eval.

Eq. 8:  E_obj = Sigmoid(Phi_o([F_obj, G_obj]))
Eq. 9:  D_hat = (2 * E_obj - 1) * D_obj + D_obj      # residual multiplicative correction,
                                                     # D_hat in (0, 2*D_obj); E=0.5 -> no change

Phi_o couples the pair by concatenating the main + guest per-member channels
(so both instances jointly determine each other's correction -- genuine
"relation" reasoning) and emitting a 2-channel error map [E_main, E_guest].

Granularity:
  * "dense"  -> E_obj is a per-cell (Hp x Wp) field (Reading D).
  * "scalar" -> E_obj is one value per instance, broadcast over the ROI so
                Eq. 9 rescales the dense D_obj uniformly (Reading H).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from instancedepth.models.phase3.candidates import PairSet


class OcclusionRelationHead(nn.Module):
    def __init__(self, per_member_channels: int, hidden_dim: int = 256,
                 num_conv: int = 3, granularity: str = "dense") -> None:
        """``per_member_channels`` = C (F_obj) + Gc (G_obj) for one instance;
        Phi_o's conv input is 2x that (main+guest concatenated)."""
        super().__init__()
        assert granularity in ("dense", "scalar")
        self.granularity = granularity
        in_ch = 2 * per_member_channels

        layers = [nn.Conv2d(in_ch, hidden_dim, 1), nn.GELU()]
        for _ in range(max(num_conv - 2, 0)):
            layers += [nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Conv2d(hidden_dim, 2, 1)   # [E_main, E_guest]

        # Zero-init the output conv so E_obj starts at sigmoid(0)=0.5 -> Eq. 9
        # is identity at init (D_hat == D_obj): Phase 3 begins as a no-op on
        # Phase-1 depth and learns corrections away from there. [Reasonable Assumption]
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, f_obj: torch.Tensor, g_obj: torch.Tensor,
                d_obj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        f_obj : (P,2,C,Hp,Wp)   g_obj : (P,2,Gc,Hp,Wp)   d_obj : (P,2,1,Hp,Wp)
        returns e_obj (P,2,1,Hp,Wp), d_hat (P,2,1,Hp,Wp)
        """
        P, _, _, Hp, Wp = f_obj.shape
        if P == 0:
            z = f_obj.new_zeros((0, 2, 1, Hp, Wp))
            return z, z

        x = torch.cat([f_obj, g_obj], dim=2)                 # (P,2,C+Gc,Hp,Wp)
        x = x.reshape(P, 2 * x.shape[2], Hp, Wp)             # concat main|guest -> (P,2M,Hp,Wp)
        feat = self.trunk(x)                                 # (P,hidden,Hp,Wp)
        e = self.out(feat)                                   # (P,2,Hp,Wp) logits

        if self.granularity == "scalar":
            e = e.mean(dim=(-2, -1), keepdim=True)           # (P,2,1,1) pooled per instance
            e = e.expand(P, 2, Hp, Wp)
        e_obj = e.sigmoid().unsqueeze(2)                     # (P,2,1,Hp,Wp)

        d_hat = (2.0 * e_obj - 1.0) * d_obj + d_obj          # Eq. 9
        return e_obj, d_hat


# --------------------------------------------------------------------------- #
def roi_masked_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean of ``values`` (P,2,1,Hp,Wp) over ``weight`` (P,2,1,Hp,Wp)
    -> (P,2). Used to reduce D_hat to a scalar per instance for L_dist / the
    reported refined depth layer. Empty weight -> 0."""
    num = (values * weight).flatten(2).sum(-1)               # (P,2)
    den = weight.flatten(2).sum(-1).clamp_min(1e-6)
    return num / den


@torch.no_grad()
def composite_refined_depth(
    base_depth: torch.Tensor,       # (B,1,H,W) Phase-1 depth_final (full resolution)
    pairs: PairSet,
    e_obj: torch.Tensor,            # (P,2,1,Hp,Wp) Eq. 8's relative-error field
    refined_layers: torch.Tensor,   # (P,2) scalar refined depth per member (mask-mean of D_hat)
    mask_prob: torch.Tensor,        # (B,N,H,W) Phase-2 mask probs
    binarize_thresh: float,
    ratio_mode: str = "dense",      # "dense" | "scalar"
    feather_px: int = 0,            # soft-blend width at mask boundaries (0 = hard edge)
) -> torch.Tensor:
    """Composite the refinement into the dense map as a RATIO field.

    Eq. 9 simplifies algebraically: D_hat = (2E-1)*D + D = 2E * D -- the
    refinement is a multiplicative correction ratio (2E) on the base depth.
    So compositing applies the upsampled RATIO to the FULL-RESOLUTION base
    depth inside each instance's mask, instead of pasting the low-resolution
    ROI depth itself (defect D1, docs/PHASE3_DIAGNOSIS.md). Properties:

      * E = 0.5 everywhere  ->  refined == base EXACTLY (true identity);
      * the base map's fine within-person geometry is preserved and merely
        modulated, never replaced by a 28x28 blur.

    Each unique instance is written ONCE: all its pair appearances are
    aggregated into one mean ratio field (defect D2), and cross-instance
    contention is arbitrated by per-instance scalar LAYER depth --
    nearest-layer-wins, mirroring data_engine/annotate.py::_flatten_id_map
    (defect D3) -- not by per-pixel value comparison.

    ratio_mode="scalar" collapses each instance's ratio to its single
    masked-mean value: maximal within-person coherence, all remaining depth
    variation comes from the base geometry. Compositing is not
    paper-specified ([Strongly Inferred] glue), so this is a legitimate
    implementation switch; the training loss always uses the dense field.

    feather_px > 0 ramps the correction in over that many pixels at the mask
    boundary (box-blurred mask as blend alpha) instead of switching on at a
    hard edge -- a ratio != 1 hard edge otherwise shows as a ring/outline
    around instances in the depth colormap.
    """
    assert ratio_mode in ("dense", "scalar")
    B, _, H, W = base_depth.shape
    refined = base_depth.clone()
    P = len(pairs)
    if P == 0:
        return refined

    # ---- aggregate every unique (batch, query) instance across its pairs ----
    inst: dict = {}   # (b, q) -> {"ratios": [(1,Hp,Wp)...], "layers": [float...], "box": (4,)}
    for p in range(P):
        b = int(pairs.batch_index[p])
        for k in range(2):
            q = int(pairs.query_idx[p, k])
            rec = inst.setdefault((b, q), {"ratios": [], "layers": [],
                                           "box": pairs.boxes_norm[p, k]})
            rec["ratios"].append(2.0 * e_obj[p, k])          # Eq. 9's ratio = 2E
            rec["layers"].append(float(refined_layers[p, k]))

    layer_buf = base_depth.new_full((B, 1, H, W), float("inf"))
    for (b, q), rec in inst.items():
        ratio_roi = torch.stack(rec["ratios"]).mean(0)       # (1,Hp,Wp)
        layer = sum(rec["layers"]) / len(rec["layers"])
        x1, y1, x2, y2 = rec["box"].tolist()
        X1, Y1 = int(round(x1 * W)), int(round(y1 * H))
        X2, Y2 = int(round(x2 * W)), int(round(y2 * H))
        X2, Y2 = max(X2, X1 + 1), max(Y2, Y1 + 1)

        region_mask = mask_prob[b, q, Y1:Y2, X1:X2] >= binarize_thresh
        if not bool(region_mask.any()):
            continue
        ratio = F.interpolate(ratio_roi[None], size=(Y2 - Y1, X2 - X1),
                              mode="bilinear", align_corners=False)[0, 0]
        if ratio_mode == "scalar":
            ratio = torch.full_like(ratio, float(ratio[region_mask].mean()))

        base_crop = base_depth[b, 0, Y1:Y2, X1:X2]
        cand = ratio * base_crop                             # 2E * D at full resolution
        if feather_px > 0:
            # box-blur the binary mask into a soft alpha: the correction ramps
            # in over ~feather_px pixels instead of switching at a hard edge
            k = 2 * feather_px + 1
            alpha = F.avg_pool2d(region_mask.float()[None, None], k, stride=1,
                                 padding=feather_px, count_include_pad=False)[0, 0]
            alpha = alpha * region_mask.float()              # never write outside the mask
            cand = alpha * cand + (1.0 - alpha) * base_crop
        cur_layer = layer_buf[b, 0, Y1:Y2, X1:X2]
        win = region_mask & (layer < cur_layer)              # nearest-LAYER-wins (per instance)
        refined[b, 0, Y1:Y2, X1:X2] = torch.where(win, cand, refined[b, 0, Y1:Y2, X1:X2])
        layer_buf[b, 0, Y1:Y2, X1:X2] = torch.where(win, torch.full_like(cur_layer, layer), cur_layer)
    return refined
