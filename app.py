#!/usr/bin/env python3
"""
AI Object Remover
-----------------
Removes objects from images using:
  • SAM (Segment Anything Model) – box-guided object selection for delete / inpaint
  • LaMa – deep learning inpainting for seamless background fill
  • rembg (BiRefNet / ISNet / U²-Net) – remove background + Keep-region tight crop

Usage:
    python app.py [image_path]
    -- or call remove_object(path) from code --
"""

import os
import sys
import threading
import traceback
import io
import base64

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

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

# Button palette: (active_bg, active_fg, hover_bg, disabled_fg, disabled_bg)
_BTN_REMOVE_BG = ("#0891b2", "#ecfeff", "#0e7490", "#67e8f9", "#0c2a32")
_BTN_KEEP      = ("#d97706", "#fffbeb", "#b45309", "#fcd34d", "#3a2607")
_BTN_DELETE    = ("#e11d48", "#fff1f2", "#be123c", "#fb7185", "#3f1519")
_BTN_UNDO      = ("#8b5cf6", "#f5f3ff", "#7c3aed", "#c4b5fd", "#2a1f3d")
_BTN_COPY      = ("#0ea5e9", "#e0f2fe", "#0284c7", "#7dd3fc", "#132734")
_BTN_SAVE      = ("#10b981", "#ecfdf5", "#059669", "#6ee7b7", "#0f2a22")

_KEEP_L_IDLE    = "📐   Keep region"
_KEEP_L_CANCEL  = "✕   Cancel"
_KEEP_L_CONFIRM = "✓   Confirm crop"

# rembg ONNX sessions — try best quality first (see rembg README “Models”).
_REMBG_MODEL_CANDIDATES: tuple[str, ...] = (
    "birefnet-general",
    "birefnet-general-lite",
    "isnet-general-use",
    "u2net",
)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _best_device() -> str:
    """Return the best available compute device: cuda > mps (Apple Silicon) > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


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


def _flatten_rgba_on_color(img: Image.Image, rgb: tuple[int, int, int]) -> Image.Image:
    """Composite RGBA onto a solid RGB background (for display or ML input)."""
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, rgb)
    bg.paste(img, mask=img.split()[3])
    return bg


def _merge_lama_into_rgba(
    orig_rgba: Image.Image, lama_rgb: Image.Image, dilated_mask_uint8: np.ndarray
) -> Image.Image:
    """Replace pixels under dilated inpaint mask with LaMa output; keep alpha elsewhere."""
    o = np.array(orig_rgba.convert("RGBA"), copy=True)
    r = np.array(lama_rgb.convert("RGB"))
    mk = dilated_mask_uint8 > 127
    o[mk, 0:3] = r[mk]
    o[mk, 3] = 255
    return Image.fromarray(o)


def _smooth_alpha_channel(rgba: Image.Image, sigma: float = 0.9) -> Image.Image:
    """Light Gaussian blur on alpha only — softer edges without touching RGB."""
    if rgba.mode != "RGBA":
        return rgba.convert("RGBA")
    r, g, b, a = rgba.split()
    a_np = np.array(a, dtype=np.float32)
    a_np = cv2.GaussianBlur(a_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    a_out = np.clip(np.round(a_np), 0, 255).astype(np.uint8)
    return Image.merge("RGBA", (r, g, b, Image.fromarray(a_out)))


def _make_fully_opaque_rgba(patch: Image.Image) -> Image.Image:
    """Tight crop as a rectangular sticker: RGB from original (transparent→white), alpha=255."""
    rgb = _flatten_rgba_on_color(patch, (255, 255, 255))
    return Image.merge("RGBA", (*rgb.split(), Image.new("L", rgb.size, 255)))


# ── Main Application ──────────────────────────────────────────────────────────

class ObjectRemoverApp:

    def __init__(self, image_path: str | None = None):
        if _DND_AVAILABLE:
            self.root = TkinterDnD.Tk()
        else:
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
        self._op_busy       = False   # async inpaint / segment / rembg
        self._rembg_lock    = threading.Lock()
        self._rembg_session = None    # lazy; shared by Remove BG + Keep region
        self._rembg_model_name: str | None = None

        # ── Edit mode: "delete" = remove highlighted object; "keep" = crop to drag box
        self.edit_mode: str = "delete"
        self.keep_confirm_pending = False
        self.pending_crop_rect: tuple[int, int, int, int] | None = None
        self._keep_tight_global: tuple[int, int, int, int] | None = None  # gx0,gy0,gx1,gy1

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

        def _btn_topbar(text, cmd, bg, hover, fg="#e0f2fe"):
            return tk.Button(
                topbar, text=text, command=cmd,
                bg=bg, fg=fg, font=("Helvetica", 12, "bold"),
                relief=tk.FLAT, padx=16, pady=6, cursor="hand2",
                activebackground=hover, activeforeground=fg, bd=0,
            )

        _btn_topbar(
            "  Add Image", self._open_dialog,
            "#2563eb", "#1d4ed8", fg="#dbeafe",
        ).pack(side=tk.LEFT, padx=12, pady=10)

        self.status_var = tk.StringVar(value="Drag an image here or click  Add Image  to start")
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

        if _DND_AVAILABLE:
            # Root + canvas both accept drops; inner widget (canvas) must bind <<Drop>>
            # or file drops on the main area are lost.
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._on_file_drop)
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind("<<Drop>>", self._on_file_drop)
            self.canvas.dnd_bind("<<DragEnter>>", self._on_drag_enter_canvas)
            self.canvas.dnd_bind("<<DragLeave>>", self._on_drag_leave_canvas)

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
            "① Drag image here  or  Add Image",
            "② Remove BG → transparent PNG",
            "③ Keep region → AI finds subject, tight crop",
            "④ Or drag to select + Delete Object",
            "⑤ Undo / Save when done",
        ):
            tk.Label(card, text=step, bg=BG_CARD, fg=TEXT_DIM,
                     font=("Helvetica", 10), wraplength=185, justify=tk.LEFT,
                     ).pack(anchor=tk.W, pady=1)

        tk.Frame(panel, bg=BG_CARD, height=1).pack(fill=tk.X, padx=10, pady=8)

        def _action_btn(text, cmd, palette):
            bg, fg, hover_bg, dis_fg, dis_bg = palette
            b = tk.Button(
                panel, text=text, command=cmd,
                bg=BG_CARD, fg=dis_fg, font=("Helvetica", 12, "bold"),
                relief=tk.FLAT, padx=12, pady=11, cursor="hand2",
                activebackground=BG_CARD, activeforeground=fg,
                disabledforeground=dis_fg, bd=1, state=tk.DISABLED,
                highlightthickness=2, highlightbackground=BG_CARD, highlightcolor=BG_CARD,
            )
            b.pack(fill=tk.X, padx=10, pady=3)
            b._active_bg = bg
            b._active_fg = fg
            b._disabled_bg = BG_CARD
            b._disabled_fg = dis_fg
            b._active_border = bg
            b._disabled_border = BG_CARD
            return b

        self.remove_bg_btn = _action_btn(
            "🎭   Remove background", self._remove_background, _BTN_REMOVE_BG
        )
        self.keep_btn = _action_btn(_KEEP_L_IDLE, self._on_keep_click, _BTN_KEEP)
        self.delete_btn = _action_btn("🗑   Delete Object", self._delete_object, _BTN_DELETE)
        self.undo_btn   = _action_btn("↩   Undo",           self._undo,           _BTN_UNDO)
        self.copy_btn   = _action_btn("📋   Copy Image",     self._copy_work_image_as_base64, _BTN_COPY)
        self.save_btn   = _action_btn("💾   Save Image",     self._save_image,     _BTN_SAVE)

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

    def _draw_placeholder(self, highlight: bool = False):
        if self.work_image is None:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            self.canvas.delete("placeholder")
            pad = 40
            border_color  = ACCENT_BLUE if highlight else "#333333"
            text_color    = ACCENT_BLUE if highlight else "#555555"
            subtext_color = ACCENT_BLUE if highlight else "#3a3a3a"
            drop_label    = "⬇  Drop to open" if highlight else "⬇  Drag an image here"
            self.canvas.create_rectangle(
                pad, pad, cw - pad, ch - pad,
                outline=border_color, dash=(10, 6), width=2, tags="placeholder",
            )
            self.canvas.create_text(
                cw // 2, ch // 2 - 18,
                text=drop_label,
                fill=text_color, font=("Helvetica", 20, "bold"), tags="placeholder",
            )
            self.canvas.create_text(
                cw // 2, ch // 2 + 16,
                text="or click  Add Image  in the toolbar above",
                fill=subtext_color, font=("Helvetica", 12), tags="placeholder",
            )

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_models_async(self):
        def _worker():
            try:
                device = _best_device()
                self._set_status_ai(f"⏳  Loading SAM  [{device}]…")
                self._load_sam()
                self._set_status_ai(f"⏳  Loading LaMa  [{device}]…")
                self._load_lama()
                self.models_ready = True
                self._set_status_ai(f"✅  AI ready  [{device}]")
                self._set_status("Drag an image here or click  Add Image  to start")
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

        device = _best_device()
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

        import torch
        self.lama = SimpleLama(device=torch.device(_best_device()))

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

    # ── Drag-and-drop handlers ─────────────────────────────────────────────────

    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    def _parse_drop_paths(self, data: str) -> list[str]:
        """Normalize paths from tkinterdnd2 (macOS file://, Tcl braces, spaces)."""
        if not data:
            return []
        try:
            raw_list = self.root.tk.splitlist(data)
        except tk.TclError:
            raw_list = [data]
        out: list[str] = []
        for raw in raw_list:
            p = (raw or "").strip()
            if not p:
                continue
            if len(p) >= 2 and p[0] == "{" and p[-1] == "}":
                p = p[1:-1]
            if p.startswith("file:"):
                from urllib.parse import unquote, urlparse

                parsed = urlparse(p)
                p = unquote(parsed.path or "")
            out.append(p)
        return out

    def _on_file_drop(self, event):
        """Handle a file dropped onto the window or canvas."""
        self._on_drag_leave_canvas(event)
        paths = self._parse_drop_paths(getattr(event, "data", "") or "")
        if not paths:
            return
        path = paths[0]
        ext  = os.path.splitext(path)[1].lower()
        if ext not in self._IMAGE_EXTS:
            messagebox.showwarning(
                "Unsupported File",
                f"Please drop an image file.\n"
                f"Supported formats: JPG, PNG, BMP, TIFF, WebP\n\n"
                f"Got: {os.path.basename(path)}",
            )
            return
        self.load_image(path)

    def _on_drag_enter_canvas(self, event):
        """Visual highlight when a file is dragged over the canvas."""
        if self.work_image is None:
            self._draw_placeholder(highlight=True)

    def _on_drag_leave_canvas(self, event):
        """Revert visual highlight when drag leaves the canvas."""
        if self.work_image is None:
            self._draw_placeholder(highlight=False)

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
            self.edit_mode = "delete"
            self.keep_confirm_pending = False
            self.pending_crop_rect = None
            self._keep_tight_global = None
            self.keep_btn.configure(text=_KEEP_L_IDLE)

            self._set_btn(self.remove_bg_btn, True)
            self._set_btn(self.keep_btn, True)
            self._set_btn(self.delete_btn, False)
            self._set_btn(self.undo_btn, False)
            self._set_btn(self.copy_btn, False)
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

        flat = _flatten_rgba_on_color(self.work_image, (17, 17, 17))
        disp = flat.resize((dw, dh), Image.LANCZOS)

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
            if self.edit_mode == "keep":
                self.keep_confirm_pending = False
                self.pending_crop_rect = None
                self._keep_tight_global = None
                self.keep_btn.configure(text=_KEEP_L_CANCEL)
                self._set_btn(self.keep_btn, True)
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

        if self.edit_mode == "keep":
            self.pending_crop_rect = (ox0, oy0, ox1, oy1)
            self._keep_detect_async(ox0, oy0, ox1, oy1)
            return

        if not self.models_ready:
            messagebox.showinfo("Please wait", "AI models are still loading.\nTry again in a moment.")
            return

        self._segment_async(np.array([ox0, oy0, ox1, oy1]))

    # ── Segmentation ───────────────────────────────────────────────────────────

    def _segment_async(self, box: np.ndarray):
        self._op_busy = True
        self._set_status("Detecting object…")
        self.progress.start(10)
        self._set_btn(self.delete_btn, False)
        self._set_btn(self.keep_btn, False)
        self._set_btn(self.remove_bg_btn, False)

        def _worker():
            try:
                img_rgb = np.array(_flatten_rgba_on_color(self.work_image, (255, 255, 255)))
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
        self._op_busy = False
        self._set_btn(self.remove_bg_btn, self.work_image is not None)

        if not mask.any():
            self._set_status("No object detected in selection — try drawing a wider box")
            self._set_btn(self.keep_btn, True)
            return
        self.mask_array   = mask
        self.has_highlight = True
        self._render(mask=mask)
        self._set_btn(self.delete_btn, True)
        self._set_btn(self.keep_btn, True)
        if self.sel_rect_id:
            self.canvas.delete(self.sel_rect_id)
        self._set_status("Object detected — click  Delete Object  to remove it")

    def _on_segment_err(self, msg: str):
        self.progress.stop()
        self._op_busy = False
        self._set_btn(self.remove_bg_btn, self.work_image is not None)
        self._set_btn(self.keep_btn, self.work_image is not None)
        self._set_status(f"Segmentation error: {msg}")
        print("Segmentation error:", msg)

    # ── rembg session (shared: Remove BG + Keep region) ─────────────────────────

    def _ensure_rembg_session(self):
        with self._rembg_lock:
            if self._rembg_session is not None:
                return self._rembg_session
            from rembg import new_session

            last_err: Exception | None = None
            for name in _REMBG_MODEL_CANDIDATES:
                try:
                    self._rembg_session = new_session(name)
                    self._rembg_model_name = name
                    return self._rembg_session
                except Exception as e:
                    last_err = e
            raise RuntimeError(
                f"Could not load any rembg model (tried {list(_REMBG_MODEL_CANDIDATES)}): {last_err}"
            )

    # ── Keep region: rembg inside user box → tight crop (opaque sticker) ────────

    def _keep_detect_async(self, ox0: int, oy0: int, ox1: int, oy1: int) -> None:
        if self.work_image is None:
            return
        self._op_busy = True
        self._set_status("Detecting subject (same model as Remove background)…")
        self.progress.start(10)
        self._set_btn(self.delete_btn, False)
        self._set_btn(self.keep_btn, False)
        self._set_btn(self.remove_bg_btn, False)

        w, h = self.work_image.size
        left = max(0, min(ox0, ox1))
        top = max(0, min(oy0, oy1))
        right = min(w, max(ox0, ox1))
        bottom = min(h, max(oy0, oy1))
        if right - left < 8 or bottom - top < 8:
            self.progress.stop()
            self._op_busy = False
            self._set_btn(self.remove_bg_btn, True)
            self._set_btn(self.keep_btn, True)
            messagebox.showwarning("Selection too small", "Draw a larger box around the subject.")
            return

        region = self.work_image.crop((left, top, right, bottom))

        def _worker():
            try:
                from rembg import remove
            except ImportError:
                self.root.after(
                    0,
                    lambda: self._on_keep_detect_err(
                        "rembg is not installed. Run: pip install rembg onnxruntime"
                    ),
                )
                return
            try:
                sess = self._ensure_rembg_session()
                rb = remove(region, session=sess).convert("RGBA")
                alpha = np.array(rb.split()[3], dtype=np.uint8)
                ys, xs = np.where(alpha > 12)
                if len(xs) == 0:
                    self.root.after(0, self._on_keep_detect_done_empty)
                    return

                pad = 8
                rx0 = max(0, int(xs.min()) - pad)
                ry0 = max(0, int(ys.min()) - pad)
                rx1 = min(region.width, int(xs.max()) + 1 + pad)
                ry1 = min(region.height, int(ys.max()) + 1 + pad)
                if rx1 - rx0 < 2 or ry1 - ry0 < 2:
                    self.root.after(0, self._on_keep_detect_done_empty)
                    return

                gx0, gy0 = left + rx0, top + ry0
                gx1, gy1 = left + rx1, top + ry1

                mask_full = np.zeros((h, w), dtype=bool)
                fg = alpha > 12
                mask_full[top:bottom, left:right] = fg

                tight = (gx0, gy0, gx1, gy1)
                self.root.after(
                    0,
                    lambda: self._on_keep_detect_done(mask_full.copy(), tight),
                )
            except Exception as e:
                self.root.after(0, lambda: self._on_keep_detect_err(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_keep_detect_done(self, mask_full: np.ndarray, tight: tuple[int, int, int, int]) -> None:
        self.progress.stop()
        self._op_busy = False
        self._set_btn(self.remove_bg_btn, True)
        self.mask_array = mask_full
        self.has_highlight = True
        self._keep_tight_global = tight
        self._render(mask=mask_full)
        self.keep_confirm_pending = True
        self.keep_btn.configure(text=_KEEP_L_CONFIRM)
        self._set_btn(self.keep_btn, True)
        self._set_btn(self.delete_btn, False)
        if self.sel_rect_id:
            self.canvas.delete(self.sel_rect_id)
        gx0, gy0, gx1, gy1 = tight
        self._set_status(
            f"Tight crop {gx1 - gx0} × {gy1 - gy0}px — original pixels inside; "
            f"outside the file is transparent when layered. Click  Confirm crop."
        )

    def _on_keep_detect_done_empty(self) -> None:
        self.progress.stop()
        self._op_busy = False
        self._set_btn(self.remove_bg_btn, True)
        self._set_btn(self.keep_btn, True)
        self.mask_array = None
        self.has_highlight = False
        self._keep_tight_global = None
        self.keep_confirm_pending = False
        self._render()
        self._set_status(
            "No subject found in that box — try a larger area or different framing."
        )

    def _on_keep_detect_err(self, msg: str) -> None:
        self.progress.stop()
        self._op_busy = False
        self._set_btn(self.remove_bg_btn, self.work_image is not None)
        self._set_btn(self.keep_btn, self.work_image is not None)
        self._keep_tight_global = None
        self.keep_confirm_pending = False
        self.mask_array = None
        self.has_highlight = False
        self._render()
        self._set_status(f"Keep-region error: {msg}")
        print("Keep-region error:", msg)
        messagebox.showerror("Keep region failed", msg)

    # ── Keep-region crop ────────────────────────────────────────────────────────

    def _reset_keep_flow_after_image_change(self) -> None:
        self.edit_mode = "delete"
        self.keep_confirm_pending = False
        self.pending_crop_rect = None
        self._keep_tight_global = None
        self.keep_btn.configure(text=_KEEP_L_IDLE)

    def _on_keep_click(self) -> None:
        if self.work_image is None or self._op_busy:
            return
        if self.keep_confirm_pending:
            self._confirm_keep_crop()
            return
        if self.edit_mode == "keep":
            self._cancel_keep_mode()
            return
        self.edit_mode = "keep"
        self.keep_confirm_pending = False
        self.pending_crop_rect = None
        if self.has_highlight:
            self.mask_array = None
            self.has_highlight = False
            self._render()
        self.keep_btn.configure(text=_KEEP_L_CANCEL)
        self._set_btn(self.keep_btn, True)
        self._set_btn(self.delete_btn, False)
        self._set_status(
            "Keep mode: drag around the subject — same AI as Remove background finds it, "
            "then crops tight. Inside the crop, colors stay original (opaque sticker).  ✕ Cancel."
        )

    def _cancel_keep_mode(self) -> None:
        self.edit_mode = "delete"
        self.keep_confirm_pending = False
        self.pending_crop_rect = None
        self._keep_tight_global = None
        self.mask_array = None
        self.has_highlight = False
        self.keep_btn.configure(text=_KEEP_L_IDLE)
        self._render()
        self._set_btn(self.keep_btn, True)
        self._set_btn(self.delete_btn, False)
        self._set_status("Keep mode cancelled.")

    def _confirm_keep_crop(self) -> None:
        if self.work_image is None or self._keep_tight_global is None:
            messagebox.showwarning(
                "Nothing to crop",
                "Drag a box on the image first so the subject can be detected.",
            )
            return
        gx0, gy0, gx1, gy1 = self._keep_tight_global
        if gx1 - gx0 < 2 or gy1 - gy0 < 2:
            messagebox.showwarning("Crop too small", "Try drawing a larger box.")
            return
        self.history.append(self.work_image.copy())
        patch = self.work_image.crop((gx0, gy0, gx1, gy1))
        self.work_image = _make_fully_opaque_rgba(patch)
        self.mask_array = None
        self.has_highlight = False
        self.sam_image_set = False
        self.orig_save_params = {"format": "PNG", "compress_level": 1}
        self._reset_keep_flow_after_image_change()
        self._render()
        self._set_btn(self.delete_btn, False)
        self._set_btn(self.keep_btn, True)
        self._set_btn(self.remove_bg_btn, True)
        self._set_btn(self.undo_btn, True)
        self._set_btn(self.copy_btn, True)
        self._set_btn(self.save_btn, True)
        self._set_status(
            f"Tight crop {gx1 - gx0} × {gy1 - gy0}px — opaque patch; layer over transparent bg."
        )

    # ── Remove background (RGBA) ───────────────────────────────────────────────

    def _remove_background(self) -> None:
        if self.work_image is None or self._op_busy:
            return
        self._op_busy = True
        self._set_status("Removing background (first run may download model weights)…")
        self.progress.start(10)
        self._set_btn(self.remove_bg_btn, False)
        self._set_btn(self.delete_btn, False)
        self._set_btn(self.keep_btn, False)

        snap = self.work_image.copy()

        def _worker():
            try:
                from rembg import remove
            except ImportError:
                self.root.after(
                    0,
                    lambda: self._on_rembg_err(
                        "rembg is not installed. Run: pip install rembg onnxruntime"
                    ),
                )
                return
            try:
                sess = self._ensure_rembg_session()
                out = remove(snap, session=sess).convert("RGBA")
                out = _smooth_alpha_channel(out, sigma=0.9)
                self.root.after(0, lambda: self._on_rembg_done(snap, out))
            except Exception as e:
                self.root.after(0, lambda: self._on_rembg_err(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_rembg_done(self, snapshot: Image.Image, result: Image.Image) -> None:
        self.progress.stop()
        self._op_busy = False
        self.history.append(snapshot)
        self.work_image = result
        self.mask_array = None
        self.has_highlight = False
        self.sam_image_set = False
        self.orig_save_params = {"format": "PNG", "compress_level": 1}
        self._reset_keep_flow_after_image_change()
        self._render()
        self._set_btn(self.remove_bg_btn, True)
        self._set_btn(self.keep_btn, True)
        self._set_btn(self.delete_btn, False)
        self._set_btn(self.undo_btn, True)
        self._set_btn(self.copy_btn, True)
        self._set_btn(self.save_btn, True)
        m = self._rembg_model_name or "rembg"
        self._set_status(f"Background removed ({m}) — transparent PNG. Save as PNG to keep alpha.")

    def _on_rembg_err(self, msg: str) -> None:
        self.progress.stop()
        self._op_busy = False
        self._set_btn(self.remove_bg_btn, self.work_image is not None)
        self._set_btn(self.keep_btn, self.work_image is not None)
        self._set_btn(self.delete_btn, self.has_highlight)
        self._set_status(f"Remove background error: {msg}")
        print("Remove background error:", msg)
        messagebox.showerror("Remove background failed", msg)

    # ── Delete / Inpaint ───────────────────────────────────────────────────────

    def _delete_object(self):
        if self.mask_array is None or self.work_image is None:
            return

        self._op_busy = True
        self._set_status("Removing object with AI inpainting…")
        self.progress.start(10)
        self._set_btn(self.delete_btn, False)
        self._set_btn(self.keep_btn, False)
        self._set_btn(self.remove_bg_btn, False)

        snap = self.work_image.copy()

        def _worker():
            try:
                self.history.append(snap)

                # Dilate mask slightly for cleaner fill edges
                raw_mask = (self.mask_array * 255).astype(np.uint8)
                dilated  = _dilate_mask(raw_mask, px=10)
                mask_pil = Image.fromarray(dilated).convert("L")

                rgb_in = _flatten_rgba_on_color(snap, (255, 255, 255))
                result_rgb = self.lama(rgb_in, mask_pil).convert("RGB")
                if snap.mode == "RGBA":
                    result = _merge_lama_into_rgba(snap, result_rgb, dilated)
                else:
                    result = result_rgb
                self.root.after(0, lambda: self._on_inpaint_done(result))
            except Exception as e:
                self.root.after(0, lambda: self._on_inpaint_err(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_inpaint_done(self, result: Image.Image):
        self.progress.stop()
        self._op_busy = False
        self.work_image    = result
        self.mask_array    = None
        self.has_highlight = False
        self.sam_image_set = False   # next drag must re-encode the new image
        self._reset_keep_flow_after_image_change()
        self._render()
        self._set_btn(self.delete_btn, False)
        self._set_btn(self.remove_bg_btn, True)
        self._set_btn(self.keep_btn, True)
        self._set_btn(self.undo_btn,   True)
        self._set_btn(self.copy_btn,   True)
        self._set_btn(self.save_btn,   True)
        self._set_status("Object removed!  Drag to select another, or save the result.")

    def _on_inpaint_err(self, msg: str):
        self.progress.stop()
        self._op_busy = False
        self._set_btn(self.remove_bg_btn, self.work_image is not None)
        self._set_btn(self.keep_btn, self.work_image is not None)
        self._set_btn(self.delete_btn, self.has_highlight)
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
        self._reset_keep_flow_after_image_change()
        self._render()
        self._set_btn(self.delete_btn, False)
        self._set_btn(self.remove_bg_btn, True)
        self._set_btn(self.keep_btn, True)
        self._set_btn(self.undo_btn, bool(self.history))
        self._set_btn(self.copy_btn, True)
        self._set_btn(self.save_btn, True)
        self._set_status("Undone.")

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_image(self):
        if self.work_image is None:
            return

        if self.work_image.mode == "RGBA":
            save_params: dict = {"format": "PNG", "compress_level": 1}
            fmt = "PNG"
        else:
            save_params = dict(self.orig_save_params)
            fmt = save_params.get("format", "PNG").upper()

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
            self.work_image.save(out_path, **save_params)
            self._set_status(f"Saved: {os.path.basename(out_path)}")
            messagebox.showinfo("Saved", f"Image saved:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _copy_work_image_as_base64(self):
        if self.work_image is None:
            return
        try:
            # Use PNG for stable lossless clipboard export (works for RGB/RGBA).
            buf = io.BytesIO()
            self.work_image.save(buf, format="PNG", compress_level=1)
            b64_text = base64.b64encode(buf.getvalue()).decode("ascii")
            self.root.clipboard_clear()
            self.root.clipboard_append(b64_text)
            self.root.update_idletasks()
            self._set_status(
                f"✅ Image copied (Base64 PNG) — {self.work_image.width} × {self.work_image.height}"
            )
        except Exception as e:
            messagebox.showerror("Clipboard Error", f"Could not copy Base64:\n{e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_btn(self, btn: tk.Button, enabled: bool):
        if enabled:
            btn.configure(
                state=tk.NORMAL,
                bg=BG_CARD,
                fg=btn._active_fg,
                activebackground=BG_CARD,
                disabledforeground=getattr(btn, "_disabled_fg", "#8e8e93"),
                highlightthickness=1,
                highlightbackground=getattr(btn, "_active_border", ACCENT_BLUE),
                highlightcolor=getattr(btn, "_active_border", ACCENT_BLUE),
                bd=0,
            )
        else:
            btn.configure(
                state=tk.DISABLED,
                bg=BG_CARD,
                fg=getattr(btn, "_disabled_fg", "#5a5a5e"),
                activebackground=BG_CARD,
                highlightthickness=1,
                highlightbackground=getattr(btn, "_disabled_border", BG_CARD),
                highlightcolor=getattr(btn, "_disabled_border", BG_CARD),
                bd=0,
            )

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
