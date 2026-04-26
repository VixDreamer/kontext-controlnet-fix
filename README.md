# kontext-controlnet-fix

> **Runtime patch** for ComfyUI that fixes the shape-mismatch crash when using
> **Flux Kontext** (`ReferenceLatent`) together with any **ControlNet** node.

No core files are modified. The patch is applied automatically at startup and
is fully transparent to the rest of ComfyUI.

---

## The Bug

When Flux Kontext reference latents are active, extra tokens are appended to
the image sequence inside `Flux._forward`:

```python
img = torch.cat([img, kontext], dim=1)
# img.shape[1] grows: img_tokens  →  img_tokens + ref_tokens
```

ControlNet residual tensors (the `"input"` and `"output"` dicts) are produced
for the **original** `img_tokens` count only. The mismatch triggers a crash
during the in-place addition inside the double/single block loops:

```
RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)
              at non-singleton dimension 1
```

**Upstream fix:** [comfyanonymous/ComfyUI#9180](https://github.com/comfyanonymous/ComfyUI/pull/9180)

---

## What This Node Does

At ComfyUI startup it applies a **monkey-patch** to `Flux.forward_orig`
(in `comfy/ldm/flux/model.py`) **without touching any core file**.

The patch wraps the `control` dict with a thin proxy (`_ControlProxy`) that
pads or truncates every residual tensor along the sequence dimension to match
the current `img` length before the original loop code sees it.

| Residual size vs `img` | Action |
|------------------------|--------|
| Shorter than `img` | Zero-pad on the right |
| Longer than `img` | Truncate to `img` length |
| Equal to `img` | Pass through unchanged |

The patch is **idempotent**: if the upstream fix is already merged into your
ComfyUI build it detects this automatically (inspects `forward_orig` source for
the `"padding_size"` token) and skips patching to avoid double-wrapping.

---

## Installation

### Option A — Manual (recommended)

1. Close ComfyUI if it is running.
2. Copy the `kontext-controlnet-fix` folder into your `custom_nodes` directory:

   ```
   ComfyUI/
   └── custom_nodes/
       └── kontext-controlnet-fix/
           ├── __init__.py
           ├── pyproject.toml
           └── README.md
   ```

3. Restart ComfyUI.
4. Look for this confirmation line in the console / log:

   ```
   [kontext-controlnet-fix] Monkey-patch applied to Flux.forward_orig.
   ```

### Option B — ComfyUI Manager

Search for **"Kontext ControlNet Fix"** in the ComfyUI Manager node browser
and click **Install**, then restart ComfyUI.

> If the node does not appear yet in the Manager registry, use Option A or C.

### Option C — Git clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/vrocg/kontext-controlnet-fix
```

Then restart ComfyUI.

---

## No New Nodes in the UI

This extension is a **pure runtime patch** — it registers **no new nodes** and
adds nothing to the ComfyUI canvas. Its only effect is preventing the crash.

---

## Recommended ControlNet Parameters When Used with Flux Kontext

Combining `ReferenceLatent` with a Flux ControlNet (Union / Canny / Depth /
OpenPose) requires a bit of tuning. These ranges are a good starting point:

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `strength` | 0.5 – 0.7 | Start at 0.6. Lower = softer guidance, preserves identity better |
| `start_percent` | 0.0 | Always start from step 0 |
| `end_percent` | **0.5 – 0.65** | **Critical** — see note below |
| VAE slot (slot 7) | Required | `ControlNetApplyAdvanced` needs the VAE for Flux latent encoding |

### Why `end_percent` matters

Setting `end_percent = 1.0` (ControlNet active for the full denoising run)
causes the skeleton / depth-map structure to dominate the output, producing a
"pencil sketch" look. Setting `end_percent = 0.6` lets the ControlNet guide
the coarse structure in the first 60 % of steps, then releases so that identity,
skin texture, and hair form naturally in the remaining 40 %.

### Recommended node chain (OpenPose example)

```
LoadImage (pose photo)
  └─► DWPreprocessor
        └─► ControlNetApplyAdvanced  ←─ slot "image"

ControlNetLoader
  └─► SetUnionControlNetType ("openpose")
        └─► ControlNetApplyAdvanced  ←─ slot "control_net"

VAELoader ──────────────────────────► ControlNetApplyAdvanced  ←─ slot "vae"

CLIPTextEncode → ReferenceLatent → FluxGuidance
  └─► ControlNetApplyAdvanced  ←─ slot "positive"

CLIPTextEncode (negative)
  └─► ControlNetApplyAdvanced  ←─ slot "negative"

ControlNetApplyAdvanced (strength=0.6, start=0.0, end=0.6)
  └─► KSampler
```

---

## How It Works (Technical)

```
Flux._forward  (ComfyUI core — unmodified)
│
├── img = torch.cat([img, kontext], dim=1)   ← img.shape[1] grows
│
└── forward_orig(img, ..., control=control_dict)
       │
       └── [PATCHED WRAPPER]
              control = _ControlProxy(control_dict, img.shape[1])
              │
              ├── double-block loop calls control.get("input")
              │     └── _fit_to(residual, img_seq_len)  ← pad or truncate
              │
              └── single-block loop calls control.get("output")
                    └── _fit_to(residual, img_seq_len)  ← pad or truncate
              │
              └── original forward_orig body runs unchanged
```

The proxy intercepts `.get("input")` and `.get("output")` on the control dict
and resizes each residual tensor via `_fit_to()`. The original `forward_orig`
function body is left **completely untouched**, making the patch resilient to
future upstream changes.

---

## Compatibility

| Component | Notes |
|-----------|-------|
| ComfyUI | Tested on master `a1344238` (2025-04) and nearby builds |
| Flux Kontext | `ReferenceLatent`, offset / index / UXO reference methods |
| ControlNet | Any Flux ControlNet — Union, Tile, Canny, Depth, OpenPose |
| Python | 3.10 + |
| PyTorch | 2.x (CPU + CUDA) |

---

## Uninstalling

Delete the `kontext-controlnet-fix` folder from `custom_nodes/` and restart
ComfyUI. No other files are affected.

---

## Credits

- **Root-cause analysis & upstream fix:**
  [comfyanonymous/ComfyUI#9180](https://github.com/comfyanonymous/ComfyUI/pull/9180)
- **Monkey-patch implementation:** [vrocg](https://github.com/vrocg)

---

## License

[MIT](LICENSE)
