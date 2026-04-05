#!/usr/bin/env python3
"""
AI Object Remover
-----------------
Removes objects from images using:
  • SAM (Segment Anything Model) – precise object boundary detection
  • LaMa – deep learning inpainting for seamless background fill

Usage:
    python app.py [image_path]
    -- or call remove_object(path) from code --
"""

import os
import sys
import threading
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk

# ── Constants ────────────────────────────────────────────────────────────────

SAM_CHECKPOINT_URL  = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
SAM_MODELS_DIR      = os.path.join(os.path.expanduser("~"), ".ai_object_remover", "models")

MAX_CANVAS_W = 1150
MAX_CANVAS_H = 780

# Colors (dark theme)
BG_ROOT      = "#0d0d0d"
BG_PANEL     = "#1c1c1e"
BG_CARD      = "#2c2c2e"
BG_CANVAS    = "#111111"
ACCENT_BLUE  = "#0a84ff"
ACCENT_RED   = "#ff3b30"
ACCENT_GREEN = "#30d158"
ACCENT_AMBER = "#ff9f0a"
TEXT_PRIMARY = "#ffffff"
TEXT_DIM     = "#8e8e93"
SEL_COLOR    = "#00c8ff"


# ── Utilities ─────────────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _download_file(url: str, dest: str) -> None:
    """Download a file, bypassing macOS SSL certificate verification issues."""
    import ssl
    import urllib.request

    # macOS Python.org installs often lack bundled CA certs; try certifi first.
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        # Fall back to an unverified context (safe for downloading known model URLs)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    with opener.open(url) as response, open(dest, "wb") as out:
        chunk_size = 1 << 20  # 1 MB
        while chunk := response.read(chunk_size):
            out.write(chunk)


def _dilate_mask(mask_uint8: np.ndarray, px: int = 8) -> np.ndarray:
    """Dilate mask by `px` pixels for cleaner inpainting edges."""
    kernel = np.ones((px, px), np.uint8)
    return cv2.dilate(mask_uint8, kernel, iterations=1)


def _infer_save_params(img: Image.Image) -> dict:
    """Return PIL save kwargs that preserve the original format and quality."""
    fmt = (img.format or "PNG").upper()
    if fmt in ("JPEG", "JPG"):
        quality = img.info.get("quality", 95)
        if not isinstance(quality, int) or not (1 <= quality <= 95):
            quality = 95
        subsampling = img.info.get("subsampling", -1)
        params: dict = {"format": "JPEG", "quality": quality, "optimize": True}
        if isinstance(subsampling, int) and subsampling >= 0:
            params["subsampling"] = subsampling
        return params
    if fmt == "PNG":
        return {"format": "PNG", "compress_level": 1}   # lossless, fast
    if fmt == "WEBP":
        q = img.info.get("quality", 90)
        return {"format": "WEBP", "quality": q if isinstance(q, int) else 90}
    # Anything else → lossless PNG
    return {"format": "PNG", "compress_level": 1}


# ── Main Application ──────────────────────────────────────────────────────────

class ObjectRemoverApp:

    def __init__(self, image_path: str | None = None):
        self.root = tk.Tk()
        self.root.title("AI Object Remover")
        self.root.configure(bg=BG_ROOT)
        self.root.geometry("1440x880")
        self.root.minsize(960, 620)

        # ── Image state
        self.orig_image: Image.Image | None = None   # untouched original
        self.work_image: Image.Image | None = None   # current edit state
        self.history: list[Image.Image] = []         # for undo
        self.image_path: str | None = None
        self.orig_save_params: dict = {}

        # ── Mask / highlight state
        self.mask_array: np.ndarray | None = None    # bool (H, W), orig size
        self.has_highlight = False

        # ── Display transform (canvas coords → original image coords)
        self.scale    = 1.0    # canvas_pixel * scale = orig_pixel
        self.off_x    = 0      # canvas x of image top-left
        self.off_y    = 0      # canvas y of image top-left
        self.tk_photo: ImageTk.PhotoImage | None = None

        # ── Drag selection
        self.drag_start: tuple[int, int] | None = None
        self.sel_rect_id: int | None = None

        # ── AI models
        self.predictor  = None    # SAM SamPredictor
        self.lama       = None    # SimpleLama
        self.models_ready   = False
        self.sam_image_set  = False   # whether set_image() called for current work_image

        self._build_ui()
        self._load_models_async()

        if image_path:
            self.root.after(250, lambda: self.load_image(image_path))

        self.root.mainloop()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar
        topbar = tk.Frame(self.root, bg=BG_PANEL, height=54)
        topbar.pack(fill=tk.X)
        topbar.pack_propagate(False)

        _btn_topbar = lambda text, cmd, bg: tk.Button(
            topbar, text=text, command=cmd,
            bg=bg, fg=TEXT_PRIMARY, font=("Helvetica", 12, "bold"),
            relief=tk.FLAT, padx=16, pady=6, cursor="hand2",
            activebackground=bg, activeforeground=TEXT_PRIMARY, bd=0,
        )
        _btn_topbar("  Open Image", self._open_dialog, ACCENT_BLUE).pack(
            side=tk.LEFT, padx=12, pady=10)

        self.status_var = tk.StringVar(value="Open an image to start")
        tk.Label(topbar, textvariable=self.status_var,
                 bg=BG_PANEL, fg=TEXT_DIM, font=("Helvetica", 11)
                 ).pack(side=tk.LEFT, padx=16)

        self.ai_var = tk.StringVar(value="⏳  Loading AI models…")
        tk.Label(topbar, textvariable=self.ai_var,
                 bg=BG_PANEL, fg=ACCENT_AMBER, font=("Helvetica", 10)
                 ).pack(side=tk.RIGHT, padx=16)

        # ── Body
        body = tk.Frame(self.root, bg=BG_ROOT)
        body.pack(fill=tk.BOTH, expand=True)

        # ── Canvas area (left)
        cf = tk.Frame(body, bg=BG_ROOT)
        cf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)

        self.canvas = tk.Canvas(cf, bg=BG_CANVAS, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>",       self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # ── Right panel
        panel = tk.Frame(body, bg=BG_PANEL, width=220)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        panel.pack_propagate(False)

        # How-to card
        card = tk.Frame(panel, bg=BG_CARD, padx=14, pady=14)
        card.pack(fill=tk.X, padx=10, pady=(12, 6))
        tk.Label(card, text="How to use", bg=BG_CARD, fg=TEXT_PRIMARY,
                 font=("Helvetica", 11, "bold")).pack(anchor=tk.W)
        for step in (
            "① Open an image",
            "② Drag to select an object",
            "③ AI highlights the object",
            "④ Click  Delete Object",
            "⑤ Repeat or Save the result",
        ):
            tk.Label(card, text=step, bg=BG_CARD, fg=TEXT_DIM,
                     font=("Helvetica", 10), wraplength=185, justify=tk.LEFT,
                     ).pack(anchor=tk.W, pady=1)

        tk.Frame(panel, bg=BG_CARD, height=1).pack(fill=tk.X, padx=10, pady=8)

        def _action_btn(text, cmd, bg):
            b = tk.Button(
                panel, text=text, command=cmd,
                bg=bg, fg=TEXT_PRIMARY, font=("Helvetica", 12, "bold"),
                relief=tk.FLAT, padx=12, pady=11, cursor="hand2",
                activebackground=bg, activeforeground=TEXT_PRIMARY,
                disabledforeground="#555555", bd=0, state=tk.DISABLED,
            )
            b.pack(fill=tk.X, padx=10, pady=3)
            return b

        self.delete_btn = _action_btn("🗑   Delete Object", self._delete_object, ACCENT_RED)
        self.undo_btn   = _action_btn("↩   Undo",           self._undo,           BG_CARD)
        self.save_btn   = _action_btn("💾   Save Image",     self._save_image,     ACCENT_GREEN)

        # Progress bar
        style = ttk.Style()
        style.theme_use("default")
        style.configure("App.Horizontal.TProgressbar",
                        troughcolor=BG_CARD, background=ACCENT_BLUE, thickness=4)
        self.progress = ttk.Progressbar(panel, style="App.Horizontal.TProgressbar",
                                        mode="indeterminate", length=198)
        self.progress.pack(pady=8, padx=10)

        # Placeholder text on canvas
        self.canvas.after(100, self._draw_placeholder)

    def _draw_placeholder(self):
        if self.work_image is None:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            self.canvas.delete("placeholder")
            self.canvas.create_text(
                cw // 2, ch // 2,
                text="Open an image to start",
                fill="#333333", font=("Helvetica", 18), tags="placeholder",
            )

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_models_async(self):
        def _worker():
            try:
                self._set_status_ai("⏳  Downloading / loading SAM…")
                self._load_sam()
                self._set_status_ai("⏳  Loading LaMa inpainter…")
                self._load_lama()
                self.models_ready = True
                self._set_status_ai("✅  AI ready")
                self._set_status("Open an image or drag to select an object")
            except Exception:
                err = traceback.format_exc()
                print(err)
                self._set_status_ai("❌  Model load failed — see console")

        threading.Thread(target=_worker, daemon=True).start()

    def _load_sam(self):
        from segment_anything import sam_model_registry, SamPredictor
        import torch

        os.makedirs(SAM_MODELS_DIR, exist_ok=True)
        ckpt = os.path.join(SAM_MODELS_DIR, SAM_CHECKPOINT_NAME)

        if not os.path.exists(ckpt):
            self._set_status("Downloading SAM model (375 MB) — first run only, please wait…")
            _download_file(SAM_CHECKPOINT_URL, ckpt)

        device = "cuda" if _cuda_available() else "cpu"
        sam = sam_model_registry["vit_b"](checkpoint=ckpt)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def _load_lama(self):
        from simple_lama_inpainting import SimpleLama

        # torch.hub caches models here; pre-download to bypass macOS SSL issues.
        lama_url  = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
        lama_ckpt = os.path.join(cache_dir, "big-lama.pt")

        if not os.path.exists(lama_ckpt):
            os.makedirs(cache_dir, exist_ok=True)
            self._set_status("Downloading LaMa model (200 MB) — first run only, please wait…")
            _download_file(lama_url, lama_ckpt)

        self.lama = SimpleLama()

    # ── Image loading ──────────────────────────────────────────────────────────

    def _open_dialog(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        try:
            raw = Image.open(path)
            self.orig_save_params = _infer_save_params(raw)
            img = raw.convert("RGB")
            self.orig_image = img.copy()
            self.work_image = img.copy()
            self.history.clear()
            self.image_path = path
            self.mask_array = None
            self.has_highlight = False
            self.sam_image_set = False

            self._set_btn(self.delete_btn, False)
            self._set_btn(self.undo_btn, False)
            self._set_btn(self.save_btn, False)

            self._render()
            self._set_status(
                f"Loaded: {os.path.basename(path)}  "
                f"({img.width} × {img.height})"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image:\n{e}")

    # ── Canvas rendering ───────────────────────────────────────────────────────

    def _on_canvas_resize(self, _event):
        if self.work_image is not None:
            self._render(mask=self.mask_array if self.has_highlight else None)
        else:
            self._draw_placeholder()

    def _render(self, mask: np.ndarray | None = None):
        if self.work_image is None:
            return

        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        iw, ih = self.work_image.size

        scale_fit = min(cw / iw, ch / ih, 1.0)
        dw = int(iw * scale_fit)
        dh = int(ih * scale_fit)
        self.scale = 1.0 / scale_fit   # canvas px → orig px
        self.off_x = (cw - dw) // 2
        self.off_y = (ch - dh) // 2

        disp = self.work_image.resize((dw, dh), Image.LANCZOS)

        if mask is not None and mask.any():
            disp = self._blend_mask(disp, mask)

        self.tk_photo = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(self.off_x, self.off_y,
                                  anchor=tk.NW, image=self.tk_photo, tags="img")

    def _blend_mask(self, disp: Image.Image, mask: np.ndarray) -> Image.Image:
        """Overlay a semi-transparent highlight + cyan border on detected object."""
        dw, dh = disp.size
        # Scale mask to display size
        mask_disp = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (dw, dh), Image.NEAREST)

        # Fill layer
        overlay = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
        fill    = Image.new("RGBA", (dw, dh), (30, 144, 255, 130))
        overlay.paste(fill, mask=mask_disp)

        # Border layer (dilated - original)
        m_np      = np.array(mask_disp)
        dilated   = cv2.dilate(m_np, np.ones((3, 3), np.uint8), iterations=2)
        border    = np.clip(dilated.astype(int) - m_np.astype(int), 0, 255).astype(np.uint8)
        border_img = Image.fromarray(border)
        border_col = Image.new("RGBA", (dw, dh), (0, 220, 255, 255))
        overlay.paste(border_col, mask=border_img)

        result = Image.alpha_composite(disp.convert("RGBA"), overlay)
        return result.convert("RGB")

    # ── Mouse events ───────────────────────────────────────────────────────────

    def _c2o(self, cx: int, cy: int) -> tuple[int, int]:
        """Canvas coords → original image coords (clamped)."""
        ox = int((cx - self.off_x) * self.scale)
        oy = int((cy - self.off_y) * self.scale)
        if self.work_image:
            ox = max(0, min(ox, self.work_image.width  - 1))
            oy = max(0, min(oy, self.work_image.height - 1))
        return ox, oy

    def _on_press(self, event):
        if self.work_image is None:
            return
        self.drag_start = (event.x, event.y)
        if self.sel_rect_id:
            self.canvas.delete(self.sel_rect_id)
        # Clear previous highlight
        if self.has_highlight:
            self.mask_array   = None
            self.has_highlight = False
            self._render()
            self._set_btn(self.delete_btn, False)

    def _on_drag(self, event):
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        if self.sel_rect_id:
            self.canvas.delete(self.sel_rect_id)
        self.sel_rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline=SEL_COLOR, width=2, dash=(5, 3),
        )

    def _on_release(self, event):
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.drag_start = None

        if abs(x1 - x0) < 8 or abs(y1 - y0) < 8:
            return   # ignore tiny accidental clicks

        ox0, oy0 = self._c2o(min(x0, x1), min(y0, y1))
        ox1, oy1 = self._c2o(max(x0, x1), max(y0, y1))

        if not self.models_ready:
            messagebox.showinfo("Please wait", "AI models are still loading.\nTry again in a moment.")
            return

        self._segment_async(np.array([ox0, oy0, ox1, oy1]))

    # ── Segmentation ───────────────────────────────────────────────────────────

    def _segment_async(self, box: np.ndarray):
        self._set_status("Detecting object…")
        self.progress.start(10)
        self._set_btn(self.delete_btn, False)

        def _worker():
            try:
                img_rgb = np.array(self.work_image)   # uint8 RGB
                if not self.sam_image_set:
                    self.predictor.set_image(img_rgb)
                    self.sam_image_set = True

                masks, scores, _ = self.predictor.predict(
                    box=box, multimask_output=True)
                best = masks[int(np.argmax(scores))]  # bool (H, W)
                self.root.after(0, lambda: self._on_segment_done(best))
            except Exception as e:
                self.root.after(0, lambda: self._on_segment_err(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_segment_done(self, mask: np.ndarray):
        self.progress.stop()
        if not mask.any():
            self._set_status("No object detected in selection — try drawing a wider box")
            return
        self.mask_array   = mask
        self.has_highlight = True
        self._render(mask=mask)
        self._set_btn(self.delete_btn, True)
        if self.sel_rect_id:
            self.canvas.delete(self.sel_rect_id)
        self._set_status("Object detected — click  Delete Object  to remove it")

    def _on_segment_err(self, msg: str):
        self.progress.stop()
        self._set_status(f"Segmentation error: {msg}")
        print("Segmentation error:", msg)

    # ── Delete / Inpaint ───────────────────────────────────────────────────────

    def _delete_object(self):
        if self.mask_array is None or self.work_image is None:
            return

        self._set_status("Removing object with AI inpainting…")
        self.progress.start(10)
        self._set_btn(self.delete_btn, False)

        def _worker():
            try:
                self.history.append(self.work_image.copy())

                # Dilate mask slightly for cleaner fill edges
                raw_mask = (self.mask_array * 255).astype(np.uint8)
                dilated  = _dilate_mask(raw_mask, px=10)
                mask_pil = Image.fromarray(dilated).convert("L")

                result = self.lama(self.work_image, mask_pil)
                result = result.convert("RGB")
                self.root.after(0, lambda: self._on_inpaint_done(result))
            except Exception as e:
                self.root.after(0, lambda: self._on_inpaint_err(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_inpaint_done(self, result: Image.Image):
        self.progress.stop()
        self.work_image    = result
        self.mask_array    = None
        self.has_highlight = False
        self.sam_image_set = False   # next drag must re-encode the new image
        self._render()
        self._set_btn(self.delete_btn, False)
        self._set_btn(self.undo_btn,   True)
        self._set_btn(self.save_btn,   True)
        self._set_status("Object removed!  Drag to select another, or save the result.")

    def _on_inpaint_err(self, msg: str):
        self.progress.stop()
        self._set_status(f"Inpainting error: {msg}")
        print("Inpainting error:", msg)
        messagebox.showerror("Inpainting Failed", f"Could not remove object:\n{msg}")

    # ── Undo ──────────────────────────────────────────────────────────────────

    def _undo(self):
        if not self.history:
            return
        self.work_image    = self.history.pop()
        self.mask_array    = None
        self.has_highlight = False
        self.sam_image_set = False
        self._render()
        self._set_btn(self.delete_btn, False)
        self._set_btn(self.undo_btn, bool(self.history))
        self._set_btn(self.save_btn, True)
        self._set_status("Undone.")

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_image(self):
        if self.work_image is None:
            return

        fmt = self.orig_save_params.get("format", "PNG").upper()
        ext_map = {
            "JPEG": (".jpg",  [("JPEG image", "*.jpg *.jpeg"), ("All files", "*.*")]),
            "PNG":  (".png",  [("PNG image",  "*.png"),        ("All files", "*.*")]),
            "WEBP": (".webp", [("WebP image", "*.webp"),       ("All files", "*.*")]),
        }
        default_ext, filetypes = ext_map.get(fmt, (".png", [("Image", "*.png"), ("All files", "*.*")]))
        base     = os.path.splitext(os.path.basename(self.image_path or "output"))[0]
        initial  = f"{base}_cleaned{default_ext}"

        out_path = filedialog.asksaveasfilename(
            title="Save Image",
            initialfile=initial,
            defaultextension=default_ext,
            filetypes=filetypes,
        )
        if not out_path:
            return

        try:
            self.work_image.save(out_path, **self.orig_save_params)
            self._set_status(f"Saved: {os.path.basename(out_path)}")
            messagebox.showinfo("Saved", f"Image saved:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_btn(self, btn: tk.Button, enabled: bool):
        btn.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _set_status_ai(self, msg: str):
        self.root.after(0, lambda: self.ai_var.set(msg))


# ── Public API ────────────────────────────────────────────────────────────────

def remove_object(image_path: str) -> None:
    """
    Open the AI Object Remover GUI for the given image.

    Args:
        image_path: Absolute or relative path to the image file.

    Example:
        from app import remove_object
        remove_object("/path/to/photo.jpg")
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    ObjectRemoverApp(image_path=os.path.abspath(image_path))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        remove_object(sys.argv[1])
    else:
        ObjectRemoverApp()
