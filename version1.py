"""
In-cabin disturbance detector: YAMNet audio events + tracked face emotion.

Requires (beyond your existing deps):
    pip install tensorflow tensorflow_hub sounddevice

Audio runs on a background thread so it never stalls the video loop.
Face emotion is gated on audio by default (GATE_FACE_ON_AUDIO=True), which
keeps the 330MB ViT asleep most of the time. Set it False to always run.

If TensorFlow/sounddevice are missing or the mic fails, the audio path
disables itself and the program degrades to face-only rather than crashing.
"""

import cv2
import torch
import time
import csv
import threading
import collections
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# ===========================================================================
# CONFIG
# ===========================================================================

GATE_FACE_ON_AUDIO = True     # only run ViT when audio is interesting
AUDIO_ENABLED = True
MIC_DEVICE = None             # None = default input device

SR = 16000
WINDOW_SEC = 0.975            # YAMNet's native frame length
WINDOW_SAMPLES = int(SR * WINDOW_SEC)
BLOCK_SAMPLES = 1600          # 100ms mic callback blocks
ANALYZE_EVERY = 0.5           # seconds between YAMNet inferences

# AudioSet class names -> score threshold.
# VERIFY these against the yamnet_class_map.csv that ships with the model;
# the code prints a warning at startup for any name it cannot find.
ALERT_CLASSES = {
    "Screaming": 0.30,
    "Shout": 0.30,
    "Yell": 0.30,
    "Children shouting": 0.35,
    "Whoop": 0.45,
}
SPEECH_CLASSES = {"Speech", "Male speech, man speaking",
                  "Female speech, woman speaking", "Conversation",
                  "Narration, monologue", "Child speech, kid speaking"}
MUSIC_CLASSES = {"Music", "Musical instrument", "Singing", "Song",
                 "Background music", "Theme music", "Pop music",
                 "Radio", "Hip hop music", "Rock music"}

MUSIC_VETO = 0.35             # suppress alert if music dominates speech
AUDIO_HOLD_SECONDS = 3.0      # audio alert stays hot this long
RMS_FLOOR = 0.005             # below this the cabin is silent; skip inference

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
ALERT_THRESHOLDS = {
    'Angry':   0.50,
    'Disgust': 0.55,
    'Fear':    0.60,
    'Sad':     0.45,
}
# The old `Neutral < 0.30 -> alert` rule is OFF by default. It fired on any
# blurry or ambiguous crop and will swamp you with false positives in a car.
USE_NEUTRAL_RULE = False
NEUTRAL_THRESHOLD = 0.30

ALERT_STREAK_SECONDS = 0.3
HOLD_SECONDS = 0.5

# ===========================================================================
# AUDIO: YAMNet listener on a background thread
# ===========================================================================


class AudioAlerter:
    def __init__(self):
        self.enabled = False
        self.buf = collections.deque(maxlen=WINDOW_SAMPLES)
        self.lock = threading.Lock()
        self.alert_until = 0.0
        self.top_label = ""
        self.top_score = 0.0
        self.rms = 0.0
        self.music_suppressed = False
        self.status = "audio off"
        self._stop = threading.Event()
        self._model = None
        self._idx = {}
        self._stream = None

    def start(self):
        try:
            import tensorflow_hub as hub
            import sounddevice as sd
        except Exception as e:
            self.status = f"audio off ({type(e).__name__})"
            print(f"[audio] disabled, missing dependency: {e}")
            return

        try:
            print("[audio] loading YAMNet (first run downloads ~4MB)...")
            self._model = hub.load("https://tfhub.dev/google/yamnet/1")

            path = self._model.class_map_path().numpy().decode("utf-8")
            with open(path) as f:
                names = [row["display_name"] for row in csv.DictReader(f)]
            self._idx = {n: i for i, n in enumerate(names)}

            missing = [c for c in ALERT_CLASSES if c not in self._idx]
            if missing:
                print(f"[audio] WARNING: these class names are NOT in the "
                      f"class map and will be ignored: {missing}")
            active = [c for c in ALERT_CLASSES if c in self._idx]
            if not active:
                print("[audio] no valid alert classes; audio disabled.")
                self.status = "audio off (no classes)"
                return
            print(f"[audio] armed on: {active}")

            self._stream = sd.InputStream(
                samplerate=SR, channels=1, dtype="float32",
                blocksize=BLOCK_SAMPLES, device=MIC_DEVICE,
                callback=self._on_audio,
            )
            self._stream.start()
            threading.Thread(target=self._loop, daemon=True).start()
            self.enabled = True
            self.status = "listening"
        except Exception as e:
            self.status = f"audio off ({type(e).__name__})"
            print(f"[audio] failed to start: {e}")

    def _on_audio(self, indata, frames, time_info, status):
        with self.lock:
            self.buf.extend(indata[:, 0])

    def _snapshot(self):
        with self.lock:
            if len(self.buf) < WINDOW_SAMPLES:
                return None
            return np.array(self.buf, dtype=np.float32)

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(ANALYZE_EVERY)
            try:
                self._analyze()
            except Exception as e:
                print(f"[audio] analyze error: {e}")

    def _analyze(self):
        wav = self._snapshot()
        if wav is None:
            return
        self.rms = float(np.sqrt(np.mean(wav ** 2)))
        if self.rms < RMS_FLOOR:
            self.top_label, self.top_score = "", 0.0
            return

        scores, _, _ = self._model(wav)
        mean = scores.numpy().mean(axis=0)

        def best_of(group):
            vals = [mean[self._idx[c]] for c in group if c in self._idx]
            return float(max(vals)) if vals else 0.0

        music = best_of(MUSIC_CLASSES)
        speech = best_of(SPEECH_CLASSES)

        best_name, best_score, fired = "", 0.0, False
        for cname, thresh in ALERT_CLASSES.items():
            if cname not in self._idx:
                continue
            s = float(mean[self._idx[cname]])
            if s > best_score:
                best_name, best_score = cname, s
            if s >= thresh:
                fired = True

        self.music_suppressed = False
        if fired and music > MUSIC_VETO and music > speech:
            fired = False
            self.music_suppressed = True

        self.top_label, self.top_score = best_name, best_score
        if fired:
            self.alert_until = time.time() + AUDIO_HOLD_SECONDS

    def is_alerting(self):
        return time.time() < self.alert_until

    def hud(self):
        if not self.enabled:
            return self.status
        tag = f"rms {self.rms:.3f}"
        if self.top_label:
            tag += f" | {self.top_label} {self.top_score:.2f}"
        if self.music_suppressed:
            tag += " | music-veto"
        return tag

    def stop(self):
        self._stop.set()
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass


# ===========================================================================
# TRACKER: IoU + centroid association with aging (stable per-person IDs)
# ===========================================================================


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
    """Greedy IoU matching with centroid fallback and track aging."""

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
        track_ids = list(self.tracks.keys())
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
        matched_dets, matched_tracks, assignments = set(), set(), {}
        for score, det_idx, tid in candidates:
            if det_idx in matched_dets or tid in matched_tracks:
                continue
            matched_dets.add(det_idx)
            matched_tracks.add(tid)
            assignments[det_idx] = tid

        results = []
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

        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]
        return results


# ===========================================================================
# FUSION
# ===========================================================================


def fuse(audio_alert, face_alert_count, n_faces):
    """0 none, 1 weak, 2 alert, 3 high."""
    face_any = face_alert_count >= 1
    face_multi = n_faces > 1 and face_alert_count >= 2
    if audio_alert and face_multi:
        return 3, "AUDIO + MULTIPLE FACES"
    if audio_alert and face_any:
        return 2, "AUDIO + FACE"
    if audio_alert:
        return 2, "AUDIO"
    if face_multi:
        return 2, "MULTIPLE FACES"
    if face_any:
        return 1, "FACE ONLY"
    return 0, ""


LEVEL_STYLE = {
    1: ((0, 200, 255), "WEAK"),
    2: ((0, 0, 255), "ALERT"),
    3: ((0, 0, 255), "HIGH ALERT"),
}


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("Loading model... (first run downloads ~330MB)")
    processor = ViTImageProcessor.from_pretrained(
        'abhilash88/face-emotion-detection')
    model = ViTForImageClassification.from_pretrained(
        'abhilash88/face-emotion-detection')
    model.eval()

    detection_interval = int(
        input("Enter detection interval (3 = responsive, 10 = smooth): "))
    if not (3 <= detection_interval <= 10):
        raise SystemExit("Error: detection interval must be between 3 and 10.")

    seconds_per_detection = detection_interval / 30.0
    alert_streak_n = max(2, round(ALERT_STREAK_SECONDS / seconds_per_detection))

    audio = AudioAlerter()
    if AUDIO_ENABLED:
        audio.start()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    tracker = FaceTracker()

    cap = cv2.VideoCapture(0)
    frame_count = 0
    level_until = 0.0
    held_level, held_reason = 0, ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1080, 720))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            faces = [tuple(int(v) for v in f) for f in faces]

            tracked = tracker.update(faces)
            audio_hot = audio.is_alerting()

            # Gate the expensive ViT on audio interest.
            gate_open = (not GATE_FACE_ON_AUDIO) or (not audio.enabled) \
                or audio_hot
            run_model = (frame_count % detection_interval == 0) and gate_open

            alert_count = 0
            for tid, (x, y, w, h) in tracked:
                track = tracker.tracks[tid]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if run_model:
                    face = frame[y:y + h, x:x + w]
                    if face.size == 0:
                        continue
                    pil_img = Image.fromarray(
                        cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    inputs = processor(pil_img, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                        preds = torch.nn.functional.softmax(
                            outputs.logits, dim=-1)[0]

                    idx = torch.argmax(preds).item()
                    emotion = emotions[idx]
                    conf = preds[idx].item()

                    bar = ALERT_THRESHOLDS.get(emotion)
                    raw_alert = bar is not None and conf > bar
                    if USE_NEUTRAL_RULE and emotion == 'Neutral' \
                            and conf < NEUTRAL_THRESHOLD:
                        raw_alert = True

                    track.streak = track.streak + 1 if raw_alert else 0
                    track.label = (f"ID{tid} {emotion} ({conf:.2f})",
                                   track.streak >= alert_streak_n)

                label, is_alert = track.label
                if is_alert:
                    alert_count += 1
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255) if is_alert else (0, 255, 0), 2)

            level, reason = fuse(audio_hot, alert_count, len(tracked))
            now = time.time()
            if level > 0:
                if level >= held_level or now >= level_until:
                    held_level, held_reason = level, reason
                level_until = now + max(HOLD_SECONDS,
                                        1.0 if level >= 2 else 0.0)
            if now >= level_until:
                held_level, held_reason = 0, ""

            if held_level > 0:
                color, text = LEVEL_STYLE[held_level]
                cv2.circle(frame, (25, 25), 12, color, -1)
                cv2.putText(frame, f"{text} - {held_reason}", (45, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            hud = audio.hud()
            if GATE_FACE_ON_AUDIO and audio.enabled and not gate_open:
                hud += " | face idle"
            cv2.putText(frame, hud, (45, 700), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 0), 1)

            frame_count += 1
            cv2.imshow("Cabin Disturbance Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        audio.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
