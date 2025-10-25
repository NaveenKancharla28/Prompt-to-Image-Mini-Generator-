# app.py
# Gradio UI that wires buttons to model_loader + image_generator.

import gradio as gr
from image_generator import generate_image, load_model

def on_load_click():
    """Hook for the 'Load Model' button."""
    try:
        status = load_model()
        return status or "Model loaded."
    except Exception as e:
        return f"Error loading model: {e}"

with gr.Blocks(title="Prompt-to-Image (SDXL, M-series friendly)") as demo:
    gr.Markdown("""
    #  Prompt-to-Image Mini Generator (SDXL)
    Generate images from text prompts using Stable Diffusion XL with a memory-friendly setup.  
    **First run:** press **Load Model** (downloads weights; can take a while).
    """)

    with gr.Row():
        load_btn = gr.Button("🔄 Load Model", variant="primary")
        load_status = gr.Textbox(label="Model Status", interactive=False)
    load_btn.click(on_load_click, outputs=load_status)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A serene lake at sunset, mountains, volumetric light, film grain",
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt (optional)",
                placeholder="blurry, low quality, watermark, distorted",
                lines=2,
            )
            with gr.Row():
                width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                height = gr.Slider(256, 1024, value=512, step=64, label="Height")

            steps = gr.Slider(10, 50, value=20, step=1, label="Inference Steps (lower=faster)")
            guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
            seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)

            gen_btn = gr.Button("🎨 Generate Image", variant="primary")

        with gr.Column(scale=1):
            out_image = gr.Image(label="Generated Image", type="pil")
            out_info = gr.Textbox(label="Generation Info", interactive=False)

    gen_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, steps, guidance_scale, seed, width, height],
        outputs=[out_image, out_info],
    )

    gr.Markdown("""
    ---
    ### Tips
    - Use **512×512** or **640×640** on M-series for speed.
    - 20–30 steps are usually enough.
    - Raise **Guidance** (8–10) for stricter prompt following; lower (4–7) for creativity.
    """)

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
