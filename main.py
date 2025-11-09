# main.py

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

USE_TORCH = True
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    USE_TORCH = False

CSV_PATH = Path("names.csv")

DELTAE_THRESH = 18.0
MIN_CONTOUR_AREA = 600
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------- COLOR ENTRY ----------------
@dataclass
class ColorEntry:
    name: str
    lab: np.ndarray


# ---------------- CSV LOADER ----------------
def load_colors(csv_path: Path):
    raw = pd.read_csv(csv_path)

    raw.columns = [c.strip() for c in raw.columns]

    required = {"name", "hex", "R", "G", "B", "L", "a", "b"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    for col in ["L", "a", "b"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["L", "a", "b"])

    colors = []
    for _, r in raw.iterrows():
        colors.append(ColorEntry(
            name=str(r["name"]).strip(),
            lab=np.array([r["L"], r["a"], r["b"]], dtype=np.float32)
        ))
    return colors


# ---------------- CLASSIFIER ----------------
class SoftmaxColor:
    def __init__(self, palette):
        self.palette = palette
        self.names = [c.name for c in palette]
        self.prototypes = np.stack([c.lab for c in palette]).astype(np.float32)
        self.torch_model = None

    def train_if_possible(self):
        if not USE_TORCH:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Xs, ys = [], []
        for idx, proto in enumerate(self.prototypes):
            jitter = np.random.normal(0, 3.0, size=(300, 3)).astype(np.float32)
            X = proto + jitter
            y = np.full(300, idx, dtype=np.int64)
            Xs.append(X)
            ys.append(y)

        X = torch.from_numpy(np.vstack(Xs)).to(device)
        y = torch.from_numpy(np.concatenate(ys)).to(device)

        model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.palette))
        ).to(device)

        opt = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(400):
            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        self.torch_model = model.eval()

    def predict_proba(self, lab_vec):
        x = lab_vec.astype(np.float32).reshape(1, 3)

        if self.torch_model:
            with torch.no_grad():
                t = torch.from_numpy(x).cuda() if next(self.torch_model.parameters()).is_cuda else torch.from_numpy(x)
                logits = self.torch_model(t).cpu().numpy()[0]
        else:
            diffs = self.prototypes - x
            d2 = np.sum(diffs * diffs, axis=1)
            logits = -0.15 * d2

        m = logits.max()
        p = np.exp(logits - m)
        return p / p.sum()

    def predict(self, lab_vec):
        p = self.predict_proba(lab_vec)
        k = int(np.argmax(p))
        return self.names[k], float(p[k]), k


# ---------------- COLOR UTILS ----------------
def bgr_to_lab_float(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

def deltaE76(lab_img, lab_ref):
    diff = lab_img - lab_ref.reshape(1, 1, 3)
    return np.sqrt(np.sum(diff * diff, axis=2))


# ---------------- MAIN ----------------
def main():
    palette = load_colors(CSV_PATH)
    classifier = SoftmaxColor(palette)
    classifier.train_if_possible()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found")

    BOX_SIZE = 200  # small detection window

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        show = frame.copy()
        h, w = frame.shape[:2]

        # --- detection box (center) ---
        cx, cy = w // 2, h // 2
        half = BOX_SIZE // 2
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half

        cv2.rectangle(show, (x1, y1), (x2, y2), (255,255,255), 2)

        # --- crop only inside the box ---
        crop = frame[y1:y2, x1:x2]

        # smooth for color stability
        crop_blur = cv2.GaussianBlur(crop, (11,11), 0)

        lab_crop = bgr_to_lab_float(crop_blur)
        mean_lab = lab_crop.reshape(-1, 3).mean(axis=0)

        pred_name, pred_conf, pred_idx = classifier.predict(mean_lab)
        target_lab = classifier.prototypes[pred_idx]

        cv2.putText(show, f"In-box: {pred_name} ({pred_conf*100:.1f}%)",
                    (10, 40), FONT, 0.8, (255,255,255), 2)

        # --- LAB for full frame ---
        blur = cv2.GaussianBlur(frame, (11,11), 0)
        lab_img = bgr_to_lab_float(blur)

        # build mask only inside the box
        dE = deltaE76(lab_img, target_lab)
        mask = (dE < DELTAE_THRESH).astype(np.uint8) * 255

        temp = np.zeros_like(mask)
        temp[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        mask = temp

        # clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # find object
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)

            if area > MIN_CONTOUR_AREA:
                x, y, w2, h2 = cv2.boundingRect(cnt)
                cv2.rectangle(show, (x,y), (x+w2,y+h2), (0,255,0), 2)
                cv2.putText(show, pred_name, (x, y-10), FONT, 0.8, (0,255,0), 2)

        cv2.imshow("Color Detector", show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
