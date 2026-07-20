import cv2
import torch
import time
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

print("Loading model... (first run downloads ~330MB)")
processor = ViTImageProcessor.from_pretrained('abhilash88/face-emotion-detection')
model = ViTForImageClassification.from_pretrained('abhilash88/face-emotion-detection')
model.eval()

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

ALERT_THRESHOLDS = {
    'Angry':   0.50,
    'Disgust': 0.55,
    'Fear':    0.60,
    'Sad':     0.45,
}
NEUTRAL_THRESHOLD = 0.30

ALERT_STREAK_SECONDS = 0.3
HOLD_SECONDS = 0.5

detection_interval = int(input("Enter detection interval (3 = responsive, 10 = smooth): "))
if not (3 <= detection_interval <= 10):
    raise SystemExit("Error: detection interval must be between 3 and 10.")

seconds_per_detection = detection_interval / 30.0
ALERT_STREAK_N = max(2, round(ALERT_STREAK_SECONDS / seconds_per_detection))



def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, box):
        self.id = tid
        self.box = box
        self.missed = 0
        self.streak = 0
        self.label = ("", False)


class FaceTracker:
    """Greedy IoU matching with a centroid-distance fallback and track aging."""
    def __init__(self, iou_thresh=0.3, max_missed=15, max_center_dist=80):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.max_center_dist = max_center_dist
        self.tracks = {}
        self._next_id = 0

    @staticmethod
    def _center(b):
        return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)

    def update(self, detections):
        """detections: list of (x, y, w, h). Returns list of (track_id, box)."""
        track_ids = list(self.tracks.keys())

        # Build all candidate matches, sorted best-first.
        candidates = []
        for det_idx, det in enumerate(detections):
            for tid in track_ids:
                score = iou(det, self.tracks[tid].box)
                if score < self.iou_thresh:
                    dc = self._center(det)
                    tc = self._center(self.tracks[tid].box)
                    dist = ((dc[0] - tc[0]) ** 2 + (dc[1] - tc[1]) ** 2) ** 0.5
                    if dist <= self.max_center_dist:
                        score = 0.01
                    else:
                        continue
                candidates.append((score, det_idx, tid))

        candidates.sort(reverse=True)

        matched_dets, matched_tracks = set(), set()
        assignments = {}
        for score, det_idx, tid in candidates:
            if det_idx in matched_dets or tid in matched_tracks:
                continue
            matched_dets.add(det_idx)
            matched_tracks.add(tid)
            assignments[det_idx] = tid

        results = []
        # Update matched tracks, spawn new ones for unmatched detections.
        for det_idx, det in enumerate(detections):
            if det_idx in assignments:
                tid = assignments[det_idx]
                self.tracks[tid].box = det
                self.tracks[tid].missed = 0
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, det)
            results.append((tid, det))

        # Age / drop unmatched tracks.
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]

        return results



face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

tracker = FaceTracker()

cap = cv2.VideoCapture(0)
frame_count = 0
alert_until = 0.0
high_alert_until = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1080, 720))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = [tuple(int(v) for v in f) for f in faces]

    # Associate detections to stable track IDs BEFORE running the model,
    tracked = tracker.update(faces)

    run_model = (frame_count % detection_interval == 0)
    alert_count = 0

    for tid, (x, y, w, h) in tracked:
        track = tracker.tracks[tid]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if run_model:
            face = frame[y:y + h, x:x + w]
            if face.size == 0:
                continue
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)

            inputs = processor(pil_img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                preds = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

            idx = torch.argmax(preds).item()
            emotion = emotions[idx]
            conf = preds[idx].item()

            bar = ALERT_THRESHOLDS.get(emotion)
            raw_alert = (bar is not None and conf > bar) or \
                        (emotion == 'Neutral' and conf < NEUTRAL_THRESHOLD)

            if raw_alert:
                track.streak += 1
            else:
                track.streak = 0

            is_alert = track.streak >= ALERT_STREAK_N
            track.label = (f"ID{tid} {emotion} ({conf:.2f})", is_alert)

        label, is_alert = track.label
        if is_alert:
            alert_count += 1

        color = (0, 0, 255) if is_alert else (0, 255, 0)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if run_model:
        if alert_count >= 1:
            alert_until = time.time() + HOLD_SECONDS
        if len(tracked) > 1 and alert_count >= 2:
            high_alert_until = time.time() + HOLD_SECONDS

    now = time.time()
    if now < high_alert_until:
        cv2.circle(frame, (25, 25), 12, (0, 0, 255), -1)
        cv2.putText(frame, "HIGH ALERT - Multiple Stressed", (45, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif now < alert_until:
        cv2.circle(frame, (25, 25), 12, (0, 0, 255), -1)
        cv2.putText(frame, "Alert - Stressed", (45, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    frame_count += 1
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
