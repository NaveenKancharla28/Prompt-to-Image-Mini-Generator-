# 🎨 Prompt-to-Image Mini Generator (SDXL)

> A lightweight **Stable Diffusion XL** image generator. 
> Built and tested locally on macOS using **VS Code** and **Gradio UI**.

---

## 🧩 Project Overview

This project is a **text-to-image generator** powered by **Stable Diffusion XL (SDXL)** and the Hugging Face **Diffusers** library.  
It allows users to generate high-quality AI art directly on their Mac using a minimal setup — no GPU server or Colab required.

---

## 🖥️ Preview

### 🔹 Model Loaded Successfully
<img src="images/model_loaded.png" width="800">

### 🔹 Generated Example Output
<img src="images/generated_sample.png" width="512">

---

## 🗂️ Project Structure

| File | Description |
|------|--------------|
| **`model_loader.py`** | Handles loading and optimization of the Stable Diffusion XL model. Automatically uses **MPS** for Apple Silicon or CPU fallback with attention slicing. |
| **`image_generator.py`** | Contains the **inference logic**. It connects to the shared pipeline and runs generation using prompts, guidance scale, and inference steps. |
| **`app.py`** | The **Gradio web interface** — connects the UI to backend functions. Handles model loading, image generation, and display. |

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/prompt-to-image-sdxl.git
cd prompt-to-image-sdxl

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate safetensors gradio pillow numpy


🚀 Run the App
python app.py

🧾 Example Prompts
"A futuristic Tokyo street at night, rain reflections, cinematic lighting"
"A cozy cabin in the mountains with snow and northern lights"
"A lake surrounded by mountains during sunset"

🧰 Tech Stack

Python 3.11+

Diffusers (Hugging Face)

Stable Diffusion XL Base 1.0

Gradio

PyTorch (MPS acceleration on Apple Silicon)

🧑‍💻 Development Notes

Built on:

macOS Sonoma (Apple M4)

VS Code IDE

Virtual Environment (venv)

Tested locally with torch.backends.mps.is_available() == True






