# model_loader.py
"""
Owns the Stable Diffusion XL pipeline as a singleton and configures device.
Optimized for Apple Silicon (MPS) with CPU fallback.
"""

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# Global pipeline handle
_PIPE = None

def get_pipe():
    """Return the loaded pipeline (or None if not loaded)."""
    global _PIPE
    return _PIPE

def unload_model() -> str:
    """Free the pipeline (optional helper)."""
    global _PIPE
    if _PIPE is not None:
        try:
            _PIPE = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            return "Model unloaded."
    return "No model loaded."

def load_model() -> str:
    """
    Load SDXL pipeline with a smaller VAE and sensible defaults.
    - Uses MPS on Apple Silicon if available
    - Falls back to CPU with attention slicing/offload
    """
    global _PIPE
    if _PIPE is not None:
        return "Model already loaded."

    # 1) VAE: memory-friendly variant
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )

    # 2) Base SDXL
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # 3) Optimizations
    pipe.enable_attention_slicing(1)

    # 4) Device selection
    if torch.backends.mps.is_available():
        pipe = pipe.to("mps")                     # Apple Silicon acceleration
    else:
        pipe = pipe.to("cpu")
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass  # not critical

    _PIPE = pipe
    return "Model loaded successfully!"
