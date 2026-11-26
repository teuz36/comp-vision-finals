import csv
import time
from pathlib import Path

import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

# ---------------- SETTINGS ----------------
BASE_DIR = Path(__file__).resolve().parent          
MODEL_PATH = BASE_DIR / 'best.pt'                   
DEFAULT_CONF = 0.657                                
VALUES = {'0_01': 0.01, '0_05': 0.05, '0_10': 0.10, '0_25': 0.25, '1': 1, '5': 5, '10': 10, '20': 20}
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f'best.pt not found. Put it in: {BASE_DIR}')
model = YOLO(str(MODEL_PATH))


def label_for(name):
    return f'P{name}' if name in ('1', '5', '10', '20') else f'{name.replace("_", ".")}c'


def run_detection(bgr, conf):
    """Detect coins in one BGR image. Returns annotated RGB image, counts, total, n, avg confidence."""
    # agnostic_nms=True stops one coin being counted twice under two different labels
    pred = model.predict(bgr, conf=conf, imgsz=640, iou=0.45, agnostic_nms=True, verbose=False)[0]
    counts = {k: 0 for k in VALUES}
    scores = []
    out = bgr.copy()
    h, w = out.shape[:2]
    th = max(2, round(min(h, w) / 350))
    fs = max(0.55, min(1.3, min(h, w) / 650))

    for box, cid, sc in zip(pred.boxes.xyxy.cpu().numpy(),
                            pred.boxes.cls.cpu().numpy().astype(int),
                            pred.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        name = str(model.names[int(cid)])
        if name in counts:
            counts[name] += 1
        scores.append(float(sc))
        text = f'{label_for(name)} {float(sc):.2f}'
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), th)
        (tw, tht), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, th))
        ty = max(25, y1 - 8)
        cv2.rectangle(out, (x1, max(0, ty - tht - bl - 5)), (x1 + tw + 6, ty + bl), (0, 255, 0), -1)
        cv2.putText(out, text, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (15, 15, 15),
                    max(1, th), cv2.LINE_AA)

    total = round(sum(VALUES[k] * v for k, v in counts.items()), 2)
    n = sum(counts.values())
    avg = float(np.mean(scores)) if scores else 0.0
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), counts, total, n, avg


def breakdown_rows(counts):
    return [[label_for(k), v, f'{VALUES[k]:.2f}', f'{VALUES[k] * v:.2f}'] for k, v in counts.items() if v]


# ---------------- TAB 1: SINGLE PHOTO ----------------
def detect_single(image, conf):
    if image is None:
        return None, [], '## Please upload a photo or use the webcam.'
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated, counts, total, n, avg = run_detection(bgr, conf)
    if n == 0:
        return annotated, [], '## No coins detected\nTry a lower confidence or a clearer photo.'
    summary = f'## Total: PHP {total:.2f}\n**Coins detected:** {n}  |  **Average confidence:** {avg:.0%}'
    return annotated, breakdown_rows(counts), summary


# ---------------- TAB 2: MULTIPLE PHOTOS ----------------
def detect_many(file_list, conf):
    if not file_list:
        return [], [], '## Please upload at least one photo.', None
    gallery, rows, grand, grand_n = [], [], 0.0, 0
    csv_rows = []
    for f in file_list:
        path = Path(f if isinstance(f, str) else f.name)
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        annotated, counts, total, n, avg = run_detection(bgr, conf)
        grand += total
        grand_n += n
        gallery.append((annotated, f'{path.name}: {n} coins, PHP {total:.2f}'))
        rows.append([path.name, n, f'{total:.2f}', f'{avg:.0%}'])
        csv_rows.append({'image': path.name, 'coins': n, 'total_php': f'{total:.2f}',
                         'avg_confidence': f'{avg:.4f}',
                         **{f'count_{k}': v for k, v in counts.items()}})
    if not rows:
        return [], [], '## Could not read any of the files.', None

    csv_path = RESULTS_DIR / f'detections_{time.strftime("%Y%m%d_%H%M%S")}.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as fh:
        wr = csv.DictWriter(fh, fieldnames=csv_rows[0].keys())
        wr.writeheader()
        wr.writerows(csv_rows)

    summary = (f'## Grand total: PHP {grand:.2f}\n'
               f'**Photos:** {len(rows)}  |  **Coins detected:** {grand_n}')
    return gallery, rows, summary, str(csv_path)


# ---------------- THE APP ----------------
with gr.Blocks(title='Philippine Coin Counter') as app:
    gr.Markdown('# 🪙 PesoScan\n'
                'Detects Philippine coins with YOLOv8 and adds up their total value.')
    conf = gr.Slider(0.1, 0.95, value=DEFAULT_CONF, step=0.01,
                     label='Confidence threshold (higher = stricter, lower = finds more coins)')

    with gr.Tab('Single photo / webcam'):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(sources=['upload', 'webcam'], type='numpy', label='Coin photo')
                btn = gr.Button('Count coins', variant='primary')
            with gr.Column():
                out_img = gr.Image(label='Detected coins')
                out_sum = gr.Markdown()
                out_tbl = gr.Dataframe(headers=['Coin', 'Count', 'Value each (PHP)', 'Subtotal (PHP)'],
                                       label='Breakdown')
        btn.click(detect_single, [inp, conf], [out_img, out_tbl, out_sum])

    with gr.Tab('Multiple photos'):
        with gr.Row():
            with gr.Column():
                files_in = gr.File(file_count='multiple', file_types=['image'], label='Coin photos')
                btn2 = gr.Button('Count all photos', variant='primary')
            with gr.Column():
                sum2 = gr.Markdown()
                gal = gr.Gallery(label='Results', columns=2)
                tbl2 = gr.Dataframe(headers=['Photo', 'Coins', 'Total (PHP)', 'Avg confidence'],
                                    label='Per-photo results')
                csv_out = gr.File(label='Download results (CSV)')
        btn2.click(detect_many, [files_in, conf], [gal, tbl2, sum2, csv_out])

if __name__ == '__main__':
    app.launch()   # opens at http://127.0.0.1:7860
