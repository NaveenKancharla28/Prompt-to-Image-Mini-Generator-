# image_generator.py


from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch

# ---- Preferred: functional API from model_loader.py ----
#   def load_model() -> str
#   def get_pipe() -> Any
try:
    from model_loader import get_pipe, load_model  # type: ignore
    _USING_FUNCTION_API = True
except Exception:
    _USING_FUNCTION_API = False

if not _USING_FUNCTION_API:
    # ---- Class-based fallback with flexible method discovery ----
    from model_loader import ModelLoader  # type: ignore
    _ML = ModelLoader()

    def _call_any(loader) -> str:
        """
        Try common init method names on ModelLoader.
        Returns a status string or raises if none exist.
        """
        for name in ("load_model", "load", "initialize", "init", "setup"):
            if hasattr(loader, name) and callable(getattr(loader, name)):
                return str(getattr(loader, name)())
        # No init method found; still okay if pipe already exists
        if getattr(loader, "pipe", None) is not None:
            return "Model already loaded (pipe present)."
        raise AttributeError(
            "ModelLoader has no load/init method. Expected one of: "
            "load_model, load, initialize, init, setup."
        )

    def load_model() -> str:
        return _call_any(_ML)

    def get_pipe():
        # Prefer explicit accessor if present
        if hasattr(_ML, "get_pipe") and callable(getattr(_ML, "get_pipe")):
            return _ML.get_pipe()
        # Fallback to attribute
        return getattr(_ML, "pipe", None)

def generate_image(
    prompt: str,
    negative_prompt: str = "",
    steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = -1,
    width: int = 512,
    height: int = 512,
) -> Tuple[Image.Image, str]:
    """
    Runs text->image inference using the shared SDXL pipeline.
    Returns (PIL.Image, info_str).
    """
    pipe = get_pipe()
    if pipe is None:
        raise RuntimeError("Model not loaded. Click 'Load Model' first or call load_model().")

    # Make a generator on the same device (MPS/CPU)
    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1)

    try:
        device = getattr(pipe, "_execution_device", None) or getattr(pipe, "device", "cpu") or "cpu"
        gen = torch.Generator(device=str(device)).manual_seed(seed)
    except Exception:
        gen = torch.Generator().manual_seed(seed)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=gen,
        width=int(width),
        height=int(height),
    )

    img = out.images[0]
    info = f"Generated ✓ | Seed: {seed} | Size: {width}x{height} | Steps: {steps} | Guidance: {guidance_scale}"
    return img, info
