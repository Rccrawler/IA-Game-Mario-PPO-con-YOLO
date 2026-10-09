"""
Autoetiquetado por visión (plantillas) + entrenamiento de YOLO.

Estructura de carpetas esperada:
    templates/
        goomba/   goomba_1.png  goomba_2.png ...
        koopa/    koopa_1.png ...
        mario/    mario_small.png mario_big.png ...
        coin/ ...  pipe/ ...  block/ ...
    dataset/images/   <- salida de capture_frames.py

Uso:
    pip install ultralytics opencv-python numpy
    python label_and_train.py preview   # guarda imágenes con cajas para revisar
    python label_and_train.py label     # genera etiquetas YOLO
    python label_and_train.py train     # prepara split y entrena

Cada clase = una carpeta dentro de templates/. Los sprites volteados
(mirando a la izquierda) se generan automáticamente.
"""
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_DIR = Path("dataset/images")
TPL_DIR = Path("templates")
LBL_DIR = Path("dataset/labels_raw")
PREVIEW_DIR = Path("dataset/preview")
YOLO_DIR = Path("dataset/yolo")

THRESH = 0.95     # similitud mínima (0-1). Sube si hay falsos positivos,
                  # baja si faltan detecciones. Con máscara, un acierto
                  # exacto da ~1.0, así que conviene que sea alto.
NMS_IOU = 0.30
VAL_FRACTION = 0.15


def load_templates():
    """Carga plantillas respetando la transparencia.
    Devuelve [(clase, imagen, máscara)]: la máscara vale 1 en los píxeles
    opacos del sprite y 0 en los transparentes, así el fondo no cuenta."""
    classes = sorted(p.name for p in TPL_DIR.iterdir() if p.is_dir())
    tpls = []
    for ci, c in enumerate(classes):
        for f in sorted((TPL_DIR / c).glob("*.png")):
            raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)  # conserva el alfa
            if raw is None:
                continue
            if raw.ndim == 2:
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            if raw.shape[2] == 4:
                bgr, mask = raw[:, :, :3], raw[:, :, 3] >= 128
            else:
                print(f"AVISO: {f} no tiene transparencia; se comparará con su fondo.")
                bgr, mask = raw, np.ones(raw.shape[:2], dtype=bool)
            ys, xs = np.where(mask)
            if len(ys) == 0:
                print(f"AVISO: {f} es totalmente transparente, se ignora.")
                continue
            # recorta al rectángulo del sprite: la caja queda ajustada
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            bgr, mask = bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1]
            t = bgr.astype(np.float32)
            m = np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)
            tpls.append((ci, t, m))
            tpls.append((ci, np.ascontiguousarray(t[:, ::-1]),
                         np.ascontiguousarray(m[:, ::-1])))  # versión volteada
    return classes, tpls


def detect(img, tpls):
    img_f = img.astype(np.float32)
    boxes, scores, cls = [], [], []
    for ci, t, m in tpls:
        h, w = t.shape[:2]
        if h > img.shape[0] or w > img.shape[1]:
            continue
        # SQDIFF_NORMED admite máscara: 0 = idéntico. Lo convertimos a similitud.
        res = cv2.matchTemplate(img_f, t, cv2.TM_SQDIFF_NORMED, mask=m)
        sim = 1.0 - np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)
        ys, xs = np.where(sim >= THRESH)
        for x, y in zip(xs, ys):
            boxes.append([int(x), int(y), w, h])
            scores.append(float(sim[y, x]))
            cls.append(ci)
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, THRESH, NMS_IOU)
    return [(cls[i], *boxes[i]) for i in np.array(keep).flatten()]


def images():
    return sorted(IMG_DIR.glob("*.png"))


def preview(n=40):
    classes, tpls = load_templates()
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for path in images()[:n]:
        img = cv2.imread(str(path))
        for ci, x, y, w, h in detect(img, tpls):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(img, classes[ci], (x, max(y - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.imwrite(str(PREVIEW_DIR / path.name), img)
    print(f"Revisa las imágenes en {PREVIEW_DIR}")


def label():
    classes, tpls = load_templates()
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    (LBL_DIR / "classes.txt").write_text("\n".join(classes))
    kept = 0
    for path in images():
        img = cv2.imread(str(path))
        H, W = img.shape[:2]
        dets = detect(img, tpls)
        if not dets:  # sin detecciones: se descarta (evita falsos "fondo")
            continue
        lines = [f"{c} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                 f"{w / W:.6f} {h / H:.6f}" for c, x, y, w, h in dets]
        (LBL_DIR / f"{path.stem}.txt").write_text("\n".join(lines))
        kept += 1
    print(f"{kept} imágenes etiquetadas. Clases: {classes}")


def train():
    from ultralytics import YOLO

    classes = (LBL_DIR / "classes.txt").read_text().split("\n")
    labeled = sorted(p for p in images() if (LBL_DIR / f"{p.stem}.txt").exists())

    # Las últimas imágenes son de niveles posteriores -> validación
    # = prueba real de generalización a niveles no vistos.
    n_val = max(1, int(len(labeled) * VAL_FRACTION))
    split = {"train": labeled[:-n_val], "val": labeled[-n_val:]}

    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    for name, files in split.items():
        (YOLO_DIR / "images" / name).mkdir(parents=True)
        (YOLO_DIR / "labels" / name).mkdir(parents=True)
        for p in files:
            shutil.copy(p, YOLO_DIR / "images" / name / p.name)
            shutil.copy(LBL_DIR / f"{p.stem}.txt",
                        YOLO_DIR / "labels" / name / f"{p.stem}.txt")

    yaml = (f"path: {YOLO_DIR.resolve()}\ntrain: images/train\nval: images/val\n"
            f"names:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(classes)))
    (YOLO_DIR / "data.yaml").write_text(yaml)

    model = YOLO("yolov8n.pt")  # modelo pequeño, suficiente para sprites
    model.train(data=str(YOLO_DIR / "data.yaml"), epochs=60, imgsz=256,
                batch=16, project="runs", name="mario_detector")
    print("Pesos en runs/mario_detector/weights/best.pt")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    {"preview": preview, "label": label, "train": train}.get(
        mode, lambda: print(__doc__))()