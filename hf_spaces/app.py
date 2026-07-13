"""
app.py — HuggingFace Spaces entry point
Real-Time Object Detection & Alert System
Built with YOLOv8-World + Gradio
"""

import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
import os

# ── Constants ────────────────────────────────────────────────────────────────
CUSTOM_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "sunglasses",
    "wallet", "pen", "keys", "coffee mug", "headphones", "monitor", "shoes",
]
# Deduplicate while preserving order
seen = set()
CUSTOM_CLASSES = [x for x in CUSTOM_CLASSES if not (x in seen or seen.add(x))]

# ── Model loading ─────────────────────────────────────────────────────────────
print("Loading YOLOv8-World model...")
model = YOLO("yolov8s-world.pt")
model.set_classes(CUSTOM_CLASSES)
print(f"✓ Model loaded with {len(CUSTOM_CLASSES)} classes")

# ── Colour palette ────────────────────────────────────────────────────────────
_CLASS_COLORS: dict = {}

def _get_color(class_name: str):
    if class_name not in _CLASS_COLORS:
        h = hash(class_name) & 0xFF
        hsv = np.uint8([[[h, 220, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        _CLASS_COLORS[class_name] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return _CLASS_COLORS[class_name]


def _draw_box(frame, box, class_name, confidence):
    x1, y1, x2, y2 = map(int, box)
    color = _get_color(class_name)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner decorators
    cl = max(10, (x2 - x1) // 8)
    for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        dx = cl if cx == x1 else -cl
        dy = cl if cy == y1 else -cl
        cv2.line(frame, (cx, cy), (cx + dx, cy), color, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy), color, 3)

    # Label
    label = f" {class_name}  {confidence:.0%} "
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
    ly1 = max(y1 - th - 8, 0)
    ly2 = ly1 + th + 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, ly1), (x1 + tw + 4, ly2), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, label, (x1 + 2, ly2 - 5), font, fs, (0, 0, 0), ft, cv2.LINE_AA)
    return frame


# ── Core detection function ───────────────────────────────────────────────────
def detect_objects(image, confidence_threshold: float, iou_threshold: float):
    """Run YOLOv8 detection on a PIL image and return annotated image + results."""
    if image is None:
        return None, "⚠️ No image provided."

    t0 = time.time()

    # PIL → BGR numpy
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = model.predict(
        frame,
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False,
        imgsz=640,
    )

    detections = []
    counts: dict = {}
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf       = float(box.conf[0])
            cls_id     = int(box.cls[0])
            class_name = model.names[cls_id]
            xyxy       = box.xyxy[0].tolist()
            frame      = _draw_box(frame, xyxy, class_name, conf)
            counts[class_name] = counts.get(class_name, 0) + 1
            detections.append(f"**{class_name.capitalize()}** — {conf:.0%}")

    elapsed = time.time() - t0

    # Build summary text
    if detections:
        summary_lines = [f"⚡ Inference: `{elapsed*1000:.0f} ms`", ""]
        summary_lines += [f"✅ {d}" for d in detections]
        summary = "\n".join(summary_lines)
    else:
        summary = f"⚡ Inference: `{elapsed*1000:.0f} ms`\n\n🔍 No objects detected above {confidence_threshold:.0%} confidence."

    # BGR → RGB for Gradio
    annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated), summary


def detect_video_frame(frame, confidence_threshold: float, iou_threshold: float):
    """Process a single webcam frame (numpy array from Gradio)."""
    if frame is None:
        return None
    pil_img = Image.fromarray(frame)
    result_img, _ = detect_objects(pil_img, confidence_threshold, iou_threshold)
    return np.array(result_img) if result_img else frame


# ── Gradio UI ─────────────────────────────────────────────────────────────────
css = """
body { font-family: 'Inter', sans-serif; }
.title-text { text-align: center; font-size: 2rem; font-weight: 800;
              background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { text-align: center; color: #6b7280; margin-bottom: 1rem; }
footer { display: none !important; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Object Detection System") as demo:

    gr.HTML("""
        <div class='title-text'>🎯 Real-Time Object Detection</div>
        <p class='subtitle'>Powered by YOLOv8-World • 80+ Object Classes • Built by Rajiv Ramteke</p>
    """)

    with gr.Row():
        conf_slider = gr.Slider(0.1, 0.95, value=0.40, step=0.05, label="🎚️ Confidence Threshold")
        iou_slider  = gr.Slider(0.1, 0.95, value=0.45, step=0.05, label="🎚️ IoU Threshold")

    with gr.Tabs():

        # ── Tab 1: Image ──────────────────────────────────────────────────────
        with gr.Tab("🖼️ Image"):
            with gr.Row():
                img_input  = gr.Image(type="pil", label="Upload Image or Use Webcam",
                                      sources=["upload", "webcam"])
                img_output = gr.Image(type="pil", label="Detection Result")
            result_text = gr.Markdown(label="Detections")
            detect_btn  = gr.Button("🔍 Detect Objects", variant="primary", size="lg")

            detect_btn.click(
                fn=detect_objects,
                inputs=[img_input, conf_slider, iou_slider],
                outputs=[img_output, result_text],
            )

        # ── Tab 2: Live Webcam ────────────────────────────────────────────────
        with gr.Tab("📷 Live Webcam"):
            gr.Markdown("**Live detection** — allow camera access when prompted.")
            webcam_out = gr.Image(label="Live Detection Feed", streaming=True)
            webcam_in  = gr.Image(sources=["webcam"], streaming=True, visible=False)

            webcam_in.stream(
                fn=detect_video_frame,
                inputs=[webcam_in, conf_slider, iou_slider],
                outputs=webcam_out,
            )

            with gr.Row():
                gr.HTML("""
                    <script>
                        // Auto-show webcam feed
                        document.addEventListener('DOMContentLoaded', function() {
                            setTimeout(function() {
                                let webcamBtn = document.querySelector('button[aria-label="webcam"]');
                                if (webcamBtn) webcamBtn.click();
                            }, 500);
                        });
                    </script>
                """)

        # ── Tab 3: Video ──────────────────────────────────────────────────────
        with gr.Tab("🎥 Video File"):
            with gr.Row():
                vid_input  = gr.Video(label="Upload Video")
                vid_output = gr.Video(label="Annotated Video")
            vid_conf   = gr.Slider(0.1, 0.95, value=0.40, step=0.05, label="Confidence")
            vid_btn    = gr.Button("▶️ Process Video", variant="primary")

            def process_video(video_path, conf):
                if video_path is None:
                    return None
                cap     = cv2.VideoCapture(video_path)
                fps     = cap.get(cv2.CAP_PROP_FPS) or 25
                w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # Build output path safely for any input extension
                base, _  = os.path.splitext(video_path)
                out_path = base + "_detected.mp4"
                writer  = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result, _ = detect_objects(pil, conf, 0.45)
                    if result:
                        writer.write(cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR))

                cap.release()
                writer.release()
                return out_path

            vid_btn.click(fn=process_video, inputs=[vid_input, vid_conf], outputs=vid_output)

    # ── Examples ──────────────────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### 📌 Supported Classes")
    classes_text = " • ".join([c.capitalize() for c in CUSTOM_CLASSES[:40]]) + " • and more..."
    gr.Markdown(f"_{classes_text}_")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
