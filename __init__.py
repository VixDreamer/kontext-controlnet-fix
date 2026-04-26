"""
kontext-controlnet-fix
Monkey-patches Flux.forward_orig to fix the shape mismatch that occurs when
Flux Kontext ReferenceLatent and ControlNet are used together.

Fix source: https://github.com/comfyanonymous/ComfyUI/pull/9180

Root cause:
  When Kontext reference latents are concatenated onto the image token sequence,
  img.shape[1] grows to (img_tokens + ref_tokens). ControlNet residual tensors
  ("input" / "output" dicts) may still be sized for img_tokens only — or, in
  the opposite edge case, for the full combined sequence.  Either mismatch
  crashes the in-place addition inside the block loops.

Fix strategy:
  Wrap the `control` dict passed into forward_orig with a thin proxy that
  pads (or truncates) each residual tensor to the expected sequence length
  before the original loop code touches it.  The original slicing logic is
  left untouched so that future upstream changes don't conflict.
"""

import inspect
import logging

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Shape helper
# ──────────────────────────────────────────────────────────────────────────────

def _fit_to(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resize `tensor` along dim-1 to `target_len` by padding or truncating."""
    cur = tensor.shape[1]
    if cur == target_len:
        return tensor
    if cur < target_len:
        pad = torch.zeros(
            tensor.shape[0], target_len - cur, tensor.shape[2],
            device=tensor.device, dtype=tensor.dtype,
        )
        return torch.cat([tensor, pad], dim=1)
    return tensor[:, :target_len]          # truncate extra tokens


# ──────────────────────────────────────────────────────────────────────────────
# Control dict proxy
# ──────────────────────────────────────────────────────────────────────────────

class _ControlProxy:
    """
    Wraps the control dict so that residual tensors returned by .get() are
    already resized to match the img sequence currently inside forward_orig.

    Double blocks ("input"):
      apply to img directly  →  target = img_seq_len

    Single blocks ("output"):
      apply to img[txt_len : txt_len + add.shape[1]]
      target = img_seq_len   (img already includes txt via torch.cat at that
                               point, so we keep txt_len + img_seq_len total,
                               and the slice window is img_seq_len wide)
    """

    def __init__(self, control, img_seq_len: int):
        self._ctrl = control
        self._img_seq_len = img_seq_len

    def get(self, key, default=None):
        residuals = self._ctrl.get(key, default)
        if residuals is None:
            return None
        return [
            _fit_to(t, self._img_seq_len) if t is not None else None
            for t in residuals
        ]

    # Pass-through for any other attribute the original code may access
    def __getattr__(self, name):
        return getattr(self._ctrl, name)

    def __contains__(self, item):
        return item in self._ctrl

    def __getitem__(self, item):
        return self._ctrl[item]


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_patched(original_forward_orig):
    def patched_forward_orig(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor = None,
        control=None,
        timestep_zero_index=None,
        transformer_options={},
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        if control is not None:
            # img at this point is the raw latent sequence (before img_in
            # projection); shape[1] is the full token count including any
            # Kontext reference tokens appended by _forward.
            control = _ControlProxy(control, img.shape[1])

        return original_forward_orig(
            self, img, img_ids, txt, txt_ids, timesteps, y,
            guidance=guidance,
            control=control,
            timestep_zero_index=timestep_zero_index,
            transformer_options=transformer_options,
            attn_mask=attn_mask,
        )

    patched_forward_orig._kontext_controlnet_patched = True
    return patched_forward_orig


# ──────────────────────────────────────────────────────────────────────────────
# Patch application
# ──────────────────────────────────────────────────────────────────────────────

def _apply_patch():
    try:
        import comfy.ldm.flux.model as flux_module
    except ImportError:
        logger.warning(
            "[kontext-controlnet-fix] comfy.ldm.flux.model not found – "
            "patch skipped (non-Flux installation?)."
        )
        return

    Flux = flux_module.Flux
    orig = Flux.forward_orig

    if getattr(orig, "_kontext_controlnet_patched", False):
        logger.debug("[kontext-controlnet-fix] Already patched – nothing to do.")
        return

    # If upstream has merged the fix, skip to avoid double-patching.
    try:
        src = inspect.getsource(orig)
        if "_kontext_controlnet_patched" in src or "padding_size" in src:
            logger.info(
                "[kontext-controlnet-fix] Upstream fix already present in "
                "forward_orig – patch not applied."
            )
            return
    except (OSError, TypeError):
        pass  # can't inspect source in frozen builds; apply the patch anyway

    Flux.forward_orig = _make_patched(orig)
    logger.info(
        "[kontext-controlnet-fix] Monkey-patch applied to Flux.forward_orig. "
        "ControlNet residuals will be shape-matched to the img sequence "
        "(including Kontext reference tokens)."
    )


_apply_patch()

# ──────────────────────────────────────────────────────────────────────────────
# ComfyUI registry (no actual nodes – pure runtime patch)
# ──────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
