# AI Object Remover

Remove unwanted objects from photos using AI — similar to the Magic Eraser feature on Xiaomi 14T Pro.

**Powered by:**
- [SAM (Segment Anything Model)](https://github.com/facebookresearch/segment-anything) – precise object detection
- [LaMa](https://github.com/advimman/lama) – deep learning inpainting for seamless background fill

---

## Features

- Drag a selection box around any object
- SAM AI detects the exact object boundary and highlights it
- One-click object removal with LaMa inpainting
- Multi-step undo support
- Saves at the original quality (no compression for PNG/WebP, original JPEG quality preserved)

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
```

### 2. Install PyTorch

Visit https://pytorch.org/get-started/locally/ and follow the instructions for your platform.  
For macOS (CPU):

```bash
pip install torch torchvision
```

### 3. Install SAM

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 4. Install the remaining dependencies

```bash
pip install simple-lama-inpainting Pillow numpy opencv-python
```

### 5. Run the app

```bash
# Open the GUI with no image
python app.py

# Open directly with an image
python app.py /path/to/photo.jpg
```

Or from Python code:

```python
from app import remove_object
remove_object("/path/to/photo.jpg")
```

---

## First Run

On the first run the app will automatically download:

| Model | Size | Location |
|-------|------|----------|
| SAM ViT-B checkpoint | ~375 MB | `~/.ai_object_remover/models/` |
| LaMa weights | ~200 MB | auto-managed by `simple-lama-inpainting` |

These are cached on disk and only downloaded once.

---

## How to Use

1. **Open** an image with the *Open Image* button (or pass a path on the command line)
2. **Drag** a box around the object you want to remove
3. The AI highlights the detected object in blue — if the result is wrong, drag again
4. Click **Delete Object** — the AI fills the area seamlessly
5. Repeat for more objects if needed
6. Click **Save Image** to export (same format and quality as the original)

---

## Requirements

- Python 3.10+
- macOS, Linux, or Windows
- CUDA GPU optional (CPU works fine, slower)
