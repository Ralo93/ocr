
import io
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import fitz  # PyMuPDFlo
import streamlit as st

# Optional OCR (lazy import when enabled)
try:
    import torch
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# -------------------------
# Configs & helpers
# -------------------------
@dataclass
class DetectionConfig:
    delta: int = 5
    min_area: int = 60
    max_area: int = 1_000_000
    max_variation: float = 0.2
    min_diversity: float = 0.2
    aspect_ratio_max: float = 15.0
    aspect_ratio_min: float = 0.2
    merge_iou_threshold: float = 0.25
    padding: int = 2


@dataclass
class ClusteringConfig:
    line_y_thresh: float = 0.6   # normalized by median height
    word_gap_k: float = 0.6      # adaptive threshold = mu + k*sigma on normalized gaps


@dataclass
class OCRConfig:
    enabled: bool = False
    model_name: str = "microsoft/trocr-small-printed"
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    max_width: int = 1280
    max_height: int = 1280


def to_pil_from_pix(pix) -> Image.Image:
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return Image.fromarray(arr, mode="RGB")


def merge_bboxes(bboxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.25) -> List[Tuple[int, int, int, int]]:
    if not bboxes:
        return []
    boxes = np.array(bboxes, dtype=np.int32)
    keep = []
    used = np.zeros(len(boxes), dtype=bool)

    def iou(a, b):
        ax0, ay0, aw, ah = a; ax1, ay1 = ax0 + aw, ay0 + ah
        bx0, by0, bw, bh = b; bx1, by1 = bx0 + bw, by0 + bh
        inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
        inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
        inter = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
        a_area = aw * ah; b_area = bw * bh
        union = a_area + b_area - inter + 1e-6
        return inter / union

    for i in range(len(boxes)):
        if used[i]: 
            continue
        cur = boxes[i]
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if iou(cur, boxes[j]) >= iou_thresh:
                x0 = min(cur[0], boxes[j][0])
                y0 = min(cur[1], boxes[j][1])
                x1 = max(cur[0] + cur[2], boxes[j][0] + boxes[j][2])
                y1 = max(cur[1] + cur[3], boxes[j][1] + boxes[j][3])
                cur = np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.int32)
                used[j] = True
        used[i] = True
        keep.append(tuple(map(int, cur.tolist())))
    return keep

def detect_text_regions(img_bgr: np.ndarray, cfg: DetectionConfig) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Create MSER and set params via setters (works across OpenCV versions)
    mser = cv2.MSER_create()
    if hasattr(mser, "setDelta"):
        mser.setDelta(int(cfg.delta))
    if hasattr(mser, "setMinArea"):
        mser.setMinArea(int(cfg.min_area))
    if hasattr(mser, "setMaxArea"):
        mser.setMaxArea(int(cfg.max_area))
    if hasattr(mser, "setMaxVariation"):
        mser.setMaxVariation(float(cfg.max_variation))
    if hasattr(mser, "setMinDiversity"):
        mser.setMinDiversity(float(cfg.min_diversity))

    regions, _ = mser.detectRegions(gray)

    bboxes = []
    H, W = gray.shape[:2]
    for pts in regions:
        x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
        ar = w / max(1, h)
        if cfg.aspect_ratio_min <= ar <= cfg.aspect_ratio_max and w > 6 and h > 6:
            x = max(0, x - cfg.padding)
            y = max(0, y - cfg.padding)
            w = min(W - x, w + 2 * cfg.padding)
            h = min(H - y, h + 2 * cfg.padding)
            bboxes.append((x, y, w, h))

    bboxes = merge_bboxes(bboxes, iou_thresh=cfg.merge_iou_threshold)
    bboxes.sort(key=lambda b: (b[1], b[0]))
    return bboxes


def group_lines(bboxes: List[Tuple[int, int, int, int]], cfg: ClusteringConfig):
    if not bboxes:
        return []
    # Use median box height to normalize vertical proximity
    heights = [h for (_, _, _, h) in bboxes]
    med_h = max(1.0, float(np.median(heights)))
    lines = []
    # Greedy: assign to existing line if vertical center close
    for (x, y, w, h) in bboxes:
        cy = y + h / 2.0
        placed = False
        for line in lines:
            ly = np.mean([yy + hh / 2.0 for (_, yy, _, hh) in line])
            if abs(cy - ly) / med_h <= cfg.line_y_thresh:
                line.append((x, y, w, h))
                placed = True
                break
        if not placed:
            lines.append([(x, y, w, h)])
    # sort tokens in each line by x
    for line in lines:
        line.sort(key=lambda b: b[0])
    return lines

def adaptive_word_clusters(line, cfg):
    """
    Return (clusters, gaps, thresh) for a given line of boxes.
    - clusters: List[List[bbox]]
    - gaps:     List[float] normalized gaps between adjacent boxes
    - thresh:   float, adaptive cut (mu + k*sigma)
    Always returns a 3-tuple.
    """
    # No or single box -> one cluster, empty gaps, zero threshold
    if not line or len(line) <= 1:
        return [line] if line else [], [], 0.0

    # Normalize by median height
    heights = [h for (_, _, _, h) in line]
    med_h = float(np.median(heights)) if heights else 1.0
    if med_h <= 0:
        med_h = 1.0

    # Compute positive normalized gaps between adjacent boxes
    gaps = []
    for i in range(len(line) - 1):
        x0, y0, w0, h0 = line[i]
        x1, y1, w1, h1 = line[i + 1]
        gap = (x1 - (x0 + w0)) / med_h
        gaps.append(max(0.0, float(gap)))

    mu = float(np.mean(gaps)) if gaps else 0.0
    sigma = float(np.std(gaps)) if len(gaps) > 1 else 0.0
    thresh = mu + float(cfg.word_gap_k) * sigma

    clusters = [[line[0]]]
    for i, gap in enumerate(gaps, start=1):
        if gap > thresh:
            clusters.append([line[i]])
        else:
            clusters[-1].append(line[i])

    return clusters, gaps, thresh


@st.cache_resource(show_spinner=False)
def load_trocr(model_name: str, device: str):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    try:
        processor = TrOCRProcessor.from_pretrained(model_name)
    except Exception:
        processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model




def run_ocr_on_crops(crops: List[Image.Image], model_tuple, device: str, max_w=1280, max_h=1280) -> List[str]:
    if model_tuple is None:
        return [""] * len(crops)
    processor, model = model_tuple
    proc_imgs = []
    for img in crops:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        proc_imgs.append(img)
    with torch.inference_mode():
        inputs = processor(images=proc_imgs, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs)
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


def draw_overlays(img: Image.Image, clusters_per_line, line_boxes, ocr_texts=None, alpha=80):
    """Draw colored rectangles for clusters; optionally annotate with OCR snippets."""
    out = img.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    # consistent random but deterministic colors per cluster
    rnd = random.Random(42)

    cluster_id = 0
    for li, clusters in enumerate(clusters_per_line):
        for cl in clusters:
            cluster_id += 1
            color = (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255), alpha)
            # union bbox for the cluster
            x0 = min(b[0] for b in cl)
            y0 = min(b[1] for b in cl)
            x1 = max(b[0] + b[2] for b in cl)
            y1 = max(b[1] + b[3] for b in cl)
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0, 120), width=2)
    out = Image.alpha_composite(out, overlay).convert("RGB")
    return out


def process_pdf_pages(pdf_bytes: bytes, page_indices: List[int], dpi: int, det_cfg: DetectionConfig, clu_cfg: ClusteringConfig,
                      ocr_cfg: OCRConfig):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results = []
    ocr_model_tuple = None
    if ocr_cfg.enabled:
        ocr_model_tuple = load_trocr(ocr_cfg.model_name, ocr_cfg.device)

    for pnum in page_indices:
        page = doc[pnum]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
        img_pil = to_pil_from_pix(pix)
        img_bgr = np.array(img_pil)[:, :, ::-1].copy()

        # detection
        bboxes = detect_text_regions(img_bgr, det_cfg)

        # lines
        lines = group_lines(bboxes, clu_cfg)

        # per-line clusters
        clusters_per_line = []
        for line in lines:
            clusters, gaps, thresh = adaptive_word_clusters(line, clu_cfg)
            clusters_per_line.append(clusters)

        # OCR (optional)
        ocr_texts = None
        if ocr_cfg.enabled and len(bboxes) > 0:
            crops = [img_pil.crop((x, y, x + w, y + h)) for (x, y, w, h) in bboxes]
            ocr_texts = run_ocr_on_crops(crops, ocr_model_tuple, ocr_cfg.device, ocr_cfg.max_width, ocr_cfg.max_height)

        # draw visualization
        vis = draw_overlays(img_pil, clusters_per_line, lines, ocr_texts=ocr_texts, alpha=70)
        results.append((pnum, vis))
    doc.close()
    return results


def export_to_pdf(images: List[Image.Image], out_path: str):
    out_doc = fitz.open()
    for im in images:
        new_page = out_doc.new_page(width=im.width, height=im.height)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        rect = fitz.Rect(0, 0, im.width, im.height)
        new_page.insert_image(rect, stream=buf.getvalue())
    out_doc.save(out_path)
    out_doc.close()

# -------------------------
# Streamlit UI (with preview + arrows + live re-clustering)
# -------------------------
st.set_page_config(page_title="PDF Adaptive Clustering Visualizer", layout="wide")

st.title("PDF Adaptive Clustering Visualizer")
st.caption("Load a PDF, preview pages, navigate with arrows, and automatically re-run detection & adaptive clustering per page. Optional TrOCR OCR.")

col_in, col_param = st.columns([2, 1], gap="large")

# --- Small raster cache (per page, per DPI) to keep navigation snappy
@st.cache_data(show_spinner=False)
def rasterize_page(pdf_bytes: bytes, page_idx: int, dpi: int) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_idx]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_pil = to_pil_from_pix(pix)
        return img_pil
    finally:
        doc.close()

# Single-page processor (reuses your pipeline)
def process_single_page(pdf_bytes: bytes, page_idx: int, dpi: int,
                        det_cfg: DetectionConfig, clu_cfg: ClusteringConfig,
                        ocr_cfg: OCRConfig) -> Tuple[Image.Image, Image.Image]:
    # original raster
    original = rasterize_page(pdf_bytes, page_idx, dpi)
    img_bgr = np.array(original)[:, :, ::-1].copy()

    # detection
    bboxes = detect_text_regions(img_bgr, det_cfg)
    # lines -> clusters
    lines = group_lines(bboxes, clu_cfg)
    clusters_per_line = []
    for line in lines:
        clusters, gaps, thresh = adaptive_word_clusters(line, clu_cfg)
        clusters_per_line.append(clusters)

    # ocr (optional)
    ocr_texts = None
    if ocr_cfg.enabled and len(bboxes) > 0:
        model_tuple = load_trocr(ocr_cfg.model_name, ocr_cfg.device)
        crops = [original.crop((x, y, x + w, y + h)) for (x, y, w, h) in bboxes]
        ocr_texts = run_ocr_on_crops(crops, model_tuple, ocr_cfg.device, ocr_cfg.max_width, ocr_cfg.max_height)

    # overlay
    clustered = draw_overlays(original, clusters_per_line, lines, ocr_texts=ocr_texts, alpha=70)
    return original, clustered


with col_param:
    st.subheader("Parameters")

    dpi = st.slider("Rasterization DPI", 96, 300, 200, step=4)

    st.markdown("**Detection confidence** (higher = stricter, fewer boxes)")
    confidence = st.slider("Confidence", 0.0, 1.0, 0.5, step=0.05)

    det_cfg = DetectionConfig()
    det_cfg.min_area = int(30 + confidence * 150)                 # 30..180
    det_cfg.max_variation = float(0.35 - confidence * 0.2)        # 0.35..0.15
    det_cfg.merge_iou_threshold = 0.25
    det_cfg.padding = 2

    st.markdown("**Clustering**")
    line_y_thresh = st.slider("Line vertical proximity (× median height)", 0.2, 1.5, 0.6, step=0.05)
    word_gap_k = st.slider("Word gap sensitivity k (μ + k·σ)", 0.0, 1.5, 0.6, step=0.05)
    clu_cfg = ClusteringConfig(line_y_thresh=line_y_thresh, word_gap_k=word_gap_k)

    st.markdown("**OCR (optional)**")
    ocr_enable = st.checkbox("Enable OCR (TrOCR small)", value=False, help="Uses microsoft/trocr-small-printed (<200M).")
    ocr_model_name = st.text_input("OCR model (HF)", "microsoft/trocr-small-printed")
    ocr_cfg = OCRConfig(enabled=ocr_enable, model_name=ocr_model_name)


with col_in:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if "page_idx" not in st.session_state:
        st.session_state.page_idx = 0
    if "total_pages" not in st.session_state:
        st.session_state.total_pages = 0
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None

    if uploaded is not None:
        # store bytes to session for consistent caching keys
        st.session_state.pdf_bytes = uploaded.getvalue()
        try:
            tmp_doc = fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf")
            st.session_state.total_pages = len(tmp_doc)
            tmp_doc.close()
            # clamp current index if needed
            st.session_state.page_idx = min(st.session_state.page_idx, st.session_state.total_pages - 1)
            st.success(f"PDF loaded: {st.session_state.total_pages} pages")
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")
            st.session_state.total_pages = 0
            st.session_state.page_idx = 0
    else:
        st.info("Upload a PDF to begin.")
        st.stop()

    # --- Navigation controls
    nav_col1, nav_col2, nav_col3 = st.columns([1, 6, 1])
    with nav_col1:
        prev_disabled = st.session_state.page_idx <= 0
        if st.button("◀", disabled=prev_disabled):
            st.session_state.page_idx = max(0, st.session_state.page_idx - 1)
    with nav_col3:
        next_disabled = st.session_state.page_idx >= (st.session_state.total_pages - 1)
        if st.button("▶", disabled=next_disabled):
            st.session_state.page_idx = min(st.session_state.total_pages - 1, st.session_state.page_idx + 1)
    with nav_col2:
        # slider lets you jump, arrows are comfortable for paging
        new_idx = st.slider("Page", 0, max(st.session_state.total_pages - 1, 0), st.session_state.page_idx)
        if new_idx != st.session_state.page_idx:
            st.session_state.page_idx = new_idx

    # --- Preview + clustered (recompute on any change)
    page_idx = st.session_state.page_idx
    pdf_bytes = st.session_state.pdf_bytes

    with st.spinner(f"Processing page {page_idx}..."):
        original_img, clustered_img = process_single_page(
            pdf_bytes,
            page_idx=page_idx,
            dpi=dpi,
            det_cfg=det_cfg,
            clu_cfg=clu_cfg,
            ocr_cfg=ocr_cfg
        )

    st.subheader(f"Page {page_idx}")
    col_preview, col_clustered = st.columns(2, gap="large")
    with col_preview:
        st.markdown("**Original**")
        st.image(original_img, use_column_width=True)
    with col_clustered:
        st.markdown("**Clustered**")
        st.image(clustered_img, use_column_width=True)

    # --- Export current page quickly
    st.markdown("#### Export")
    exp_c1, exp_c2 = st.columns([1, 2])
    with exp_c1:
        if st.button("Export current annotated page as PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                out_path = tmp_file.name
            try:
                export_to_pdf([clustered_img], out_path)
                with open(out_path, "rb") as f:
                    pdf_data = f.read()
                st.download_button(
                    "Download current page",
                    data=pdf_data,
                    file_name=f"clustered_page_{page_idx}.pdf",
                    mime="application/pdf"
                )
            finally:
                try: os.unlink(out_path)
                except Exception: pass

    # --- Optional: process all pages & export (slower)
    with exp_c2:
        if st.button("Process ALL pages & export annotated PDF"):
            all_imgs = []
            with st.spinner("Processing all pages..."):
                for p in range(st.session_state.total_pages):
                    _, clustered_p = process_single_page(
                        pdf_bytes,
                        page_idx=p,
                        dpi=dpi,
                        det_cfg=det_cfg,
                        clu_cfg=clu_cfg,
                        ocr_cfg=ocr_cfg
                    )
                    all_imgs.append(clustered_p)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                out_path = tmp_file.name
            try:
                export_to_pdf(all_imgs, out_path)
                with open(out_path, "rb") as f:
                    pdf_data = f.read()
                st.download_button(
                    "Download full annotated PDF",
                    data=pdf_data,
                    file_name="clustered_output_full.pdf",
                    mime="application/pdf"
                )
            finally:
                try: os.unlink(out_path)
                except Exception: pass

