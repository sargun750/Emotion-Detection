"""
READY TO GO — In-cabin disturbance detector.

Combines two pipelines:
  * AUDIO  — YAMNet shout/scream detection, identical to `Final_audio_visual.py`
             (AudioAlerter on a background thread + the same fusion rules).
  * FACE   — MediaPipe Face Landmarker -> face alignment -> HSEmotion-ONNX
             classifier -> temporal smoothing, as in `emotion_full.py`, but with
             the classifier sampled every Nth frame like `Final_audio_visual.py`.

A full red ALERT requires BOTH a shout/scream AND a stressed facial expression.

On startup you are asked for a detection interval (3 = responsive, 10 = smooth),
exactly as in `Final.py` / `Final_audio_visual.py`. The landmarker still runs on
every frame so the box and tilt angle stay smooth; only the emotion classifier is
sampled, and its last label is reused in between.

Requirements:
    pip install tensorflow tensorflow_hub sounddevice
    pip install mediapipe opencv-python hsemotion-onnx numpy

Asset file (already copied next to this script):
    face_landmarker.task   (MediaPipe face landmarker model)

If TensorFlow/sounddevice are missing or the mic fails, the audio path disables
itself and the program degrades to face-only rather than crashing.

Run:
    python "READY TO GO.py"
Press 'q' to quit.
"""

import time
import csv
import threading
import collections
from collections import deque
from pathlib import Path

import urllib.request  # must be imported before hsemotion_onnx - works around a
                       # urllib.request attribute bug in that package on newer Python

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer


# ===========================================================================
# AUDIO CONFIG  (identical to Final_audio_visual.py)
# ===========================================================================

AUDIO_ENABLED = True
DEBUG_AUDIO = True            # print YAMNet's top-5 classes each cycle; turn off once tuned
MIC_DEVICE = None             # None = default input device

SR = 16000
WINDOW_SEC = 0.975            # YAMNet's native frame length
WINDOW_SAMPLES = int(SR * WINDOW_SEC)
BLOCK_SAMPLES = 1600          # 100ms mic callback blocks
ANALYZE_EVERY = 0.5           # seconds between YAMNet inferences

ALERT_CLASSES = {
    "Screaming": 0.05,
    "Shout": 0.05,
    "Yell": 0.15,
    "Children shouting": 0.20,
    "Whoop": 0.30,
}
SPEECH_CLASSES = {"Speech", "Male speech, man speaking",
                  "Female speech, woman speaking", "Conversation",
                  "Narration, monologue", "Child speech, kid speaking"}
MUSIC_CLASSES = {"Music", "Musical instrument", "Singing", "Song",
                 "Background music", "Theme music", "Pop music",
                 "Radio", "Hip hop music", "Rock music"}

MUSIC_VETO = 0.35
AUDIO_HOLD_SECONDS = 1.0
RMS_FLOOR = 0.002
LOUD_SPEECH_RMS = 0.16   # speech louder than this = shouting. TUNE from [audio] rms:
                         # set between your normal-talk rms and your shout rms.
SPEECH_PRESENT  = 0.20   # min speech score to treat loud audio as a raised voice (not noise)


# ===========================================================================
# FACE CONFIG  (identical to emotion_full.py)
# ===========================================================================

# resolved next to this script, so the program works from any working directory
LANDMARKER_MODEL_PATH = str(Path(__file__).resolve().parent / "face_landmarker.task")
EMOTION_MODEL_NAME = "enet_b0_8_best_afew"

LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness",
          "Neutral", "Sadness", "Surprise"]

# eye-corner landmark indices in MediaPipe's 468-point face mesh, used to
# measure head tilt and align the crop before it goes to the emotion model
LEFT_EYE_LANDMARK = 33
RIGHT_EYE_LANDMARK = 263

# temporal smoothing knobs
SMOOTHING_SECONDS = 0.5      # wall-clock span of the rolling average. Converted to a
                             # sample count once detection_interval is known, the same
                             # way Final_audio_visual.py derives its alert streak, so
                             # the smoothing feels the same at interval 3 and at 10.
                             # (A fixed sample count would silently stretch to several
                             # seconds of lag at large intervals.)
MIN_SMOOTHING_SAMPLES = 3    # floor, so the average never collapses to a single frame
MIN_HOLD_SECONDS = 1.5       # once switched, must hold this long (real time) before switching again
CONFIDENCE_THRESHOLD = 0.35  # averaged score must clear this to be eligible to switch

CROP_MARGIN = 0.4  # extra padding around the face bounding box, as a fraction of its size

# Which HSEmotion labels count as "stressed", and the averaged-confidence bar
# each must clear before it contributes a face alert. Happiness/Neutral/Surprise
# never alert. (Mirrors the ALERT_THRESHOLDS idea from Final_audio_visual.py,
# remapped to HSEmotion's 8-class label set.)
ALERT_THRESHOLDS = {
    'Anger':    0.50,
    'Contempt': 0.50,
    'Disgust':  0.55,
    'Fear':     0.60,
    'Sadness':  0.45,
}

HOLD_SECONDS = 0.5           # banner latch (fusion display)


# ===========================================================================
# AUDIO: YAMNet listener on a background thread (identical to Final_audio_visual.py)
# ===========================================================================


class AudioAlerter:
    def __init__(self):
        self.enabled = False
        self.buf = collections.deque(maxlen=WINDOW_SAMPLES)
        self.lock = threading.Lock()
        self.alert_until = 0.0
        self.top_label = ""
        self.top_score = 0.0
        self.top_overall = ""
        self.top_overall_score = 0.0
        self.rms = 0.0
        self.music_suppressed = False
        self.status = "audio off"
        self._stop = threading.Event()
        self._model = None
        self._idx = {}
        self._names = []
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
            self._names = names

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
            self.top_overall, self.top_overall_score = "", 0.0
            return

        scores, _, _ = self._model(wav)
        arr = scores.numpy()
        mean = arr.mean(axis=0)
        # A shout is transient; averaging it across the ~1s window buries it.
        # Use the per-window PEAK for alert classes so short events survive.
        peak = arr.max(axis=0)

        # Diagnostic: YAMNet's single best guess for this window, regardless of
        # our alert list. Lets you see e.g. "Speech 0.62" when a shout is being
        # classified as speech instead of firing an alert.
        top_i = int(peak.argmax())
        self.top_overall = self._names[top_i] if self._names else str(top_i)
        self.top_overall_score = float(peak[top_i])

        if DEBUG_AUDIO and self._names:
            order = peak.argsort()[::-1][:5]
            top5 = ", ".join(f"{self._names[i]} {peak[i]:.2f}" for i in order)
            print(f"[audio] rms {self.rms:.3f} | top5: {top5}")

        def best_of(group):
            vals = [mean[self._idx[c]] for c in group if c in self._idx]
            return float(max(vals)) if vals else 0.0

        music = best_of(MUSIC_CLASSES)
        speech = best_of(SPEECH_CLASSES)

        best_name, best_score, fired = "", 0.0, False
        for cname, thresh in ALERT_CLASSES.items():
            if cname not in self._idx:
                continue
            s = float(peak[self._idx[cname]])
            if s > best_score:
                best_name, best_score = cname, s
            if s >= thresh:
                fired = True

        # Normal shouting is usually labelled "Speech" by YAMNet, so also treat
        # loud, speech-dominant audio as a disturbance.
        if speech >= SPEECH_PRESENT and self.rms >= LOUD_SPEECH_RMS:
            fired = True
            if speech > best_score:
                best_name, best_score = "Loud speech", speech

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
        if self.top_overall:
            tag += f" | top: {self.top_overall} {self.top_overall_score:.2f}"
        if self.top_label:
            tag += f" | alert {self.top_label} {self.top_score:.2f}"
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
# FACE: temporal smoothing (identical to emotion_full.py)
# ===========================================================================

class EmotionSmoother:
    """
    Turns noisy per-frame emotion scores into a stable label using:
      1. An 8-frame rolling average - the decision each frame is based on the
         mean of the last `window_size` raw score vectors, not a single frame.
      2. A real-time minimum hold - once the displayed label switches, it
         cannot switch again until `min_hold_seconds` of wall-clock time have
         passed, regardless of frame rate. This is deliberately time-based
         (not frame-count-based) so the hold duration stays consistent even
         if the pipeline's fps varies (e.g. slower on a phone than a desktop).
    """

    def __init__(self, labels, window_size=8, min_hold_seconds=1.5, confidence_threshold=0.35):
        self.labels = labels
        self.window_size = window_size
        self.min_hold_seconds = min_hold_seconds
        self.confidence_threshold = confidence_threshold

        self.buffer = deque(maxlen=window_size)
        self.stable_label = "Neutral"
        self.last_switch_time = time.time()

    def update(self, raw_scores):
        self.buffer.append(raw_scores)

        # wait until we actually have a full window before making decisions
        if len(self.buffer) < self.window_size:
            return self.stable_label, np.mean(self.buffer, axis=0)

        avg_scores = np.mean(self.buffer, axis=0)
        top_idx = int(np.argmax(avg_scores))
        top_label = self.labels[top_idx]
        top_conf = avg_scores[top_idx]

        now = time.time()
        time_since_switch = now - self.last_switch_time

        can_switch = (
            top_label != self.stable_label
            and top_conf >= self.confidence_threshold
            and time_since_switch >= self.min_hold_seconds
        )

        if can_switch:
            self.stable_label = top_label
            self.last_switch_time = now

        return self.stable_label, avg_scores


# ===========================================================================
# FACE: alignment (identical to emotion_full.py)
# ===========================================================================

def align_and_crop(frame, landmarks, frame_w, frame_h, margin=CROP_MARGIN):
    """
    Uses eye-corner landmarks to (a) measure head tilt and (b) rotate the
    frame so the face is level before cropping. Fixes the emotion model
    losing accuracy on tilted heads.
    """
    pts = np.array([(lm.x * frame_w, lm.y * frame_h) for lm in landmarks])

    left_eye = pts[LEFT_EYE_LANDMARK]
    right_eye = pts[RIGHT_EYE_LANDMARK]

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    box_w = (x_max - x_min) * (1 + margin)
    box_h = (y_max - y_min) * (1 + margin)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame_w, frame_h))

    x1, y1 = int(cx - box_w / 2), int(cy - box_h / 2)
    x2, y2 = int(cx + box_w / 2), int(cy + box_h / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)

    face_crop = rotated_frame[y1:y2, x1:x2]
    return face_crop, (x1, y1, x2, y2), angle


# ===========================================================================
# FUSION  (identical to Final_audio_visual.py)
# ===========================================================================


def fuse(audio_alert, face_alert_count, n_faces):
    """Full ALERT requires BOTH a shout/scream AND a stressed face.
    A single signal on its own is only a weak (informational) indicator.
    0 none, 1 weak (one signal), 2 alert (audio + face), 3 high."""
    face_any = face_alert_count >= 1
    face_multi = n_faces > 1 and face_alert_count >= 2
    if audio_alert and face_multi:
        return 3, "SHOUT + MULTIPLE STRESSED FACES"
    if audio_alert and face_any:
        return 2, "SHOUT + STRESSED FACE"
    if audio_alert:
        return 1, "SHOUT ONLY"
    if face_any:
        return 1, "STRESSED FACE ONLY"
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
    print("Loading models...")
    emotion_recognizer = HSEmotionRecognizer(model_name=EMOTION_MODEL_NAME)

    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH)
    landmarker_options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)

    detection_interval = int(
        input("Enter detection interval (3 = responsive, 10 = smooth): "))
    if not (3 <= detection_interval <= 10):
        raise SystemExit("Error: detection interval must be between 3 and 10.")

    # Same 30fps assumption Final_audio_visual.py makes when sizing its streak.
    seconds_per_detection = detection_interval / 30.0
    window_size = max(MIN_SMOOTHING_SAMPLES,
                      round(SMOOTHING_SECONDS / seconds_per_detection))

    smoother = EmotionSmoother(
        LABELS,
        window_size=window_size,
        min_hold_seconds=MIN_HOLD_SECONDS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    audio = AudioAlerter()
    if AUDIO_ENABLED:
        audio.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print(f"Running: classifier every {detection_interval} frames, "
          f"averaging {window_size} samples. Press 'q' to quit.")
    frame_count = 0
    level_until = 0.0
    held_level, held_reason = 0, ""
    # Last classifier result, reused on the frames where it does not run.
    face_label = ("Neutral", 0.0, False)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            audio_hot = audio.is_alerting()
            face_alert_count = 0
            n_faces = 0
            run_model = (frame_count % detection_interval == 0)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                face_crop, box, angle = align_and_crop(frame, landmarks, frame_w, frame_h)

                if face_crop.size > 0:
                    n_faces = 1

                    if run_model:
                        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        _raw_emotion, raw_scores = emotion_recognizer.predict_emotions(
                            face_crop_rgb, logits=False
                        )
                        stable_emotion, avg_scores = smoother.update(raw_scores)

                        conf = float(avg_scores[LABELS.index(stable_emotion)])
                        bar = ALERT_THRESHOLDS.get(stable_emotion)
                        face_label = (stable_emotion, conf,
                                      bar is not None and conf >= bar)

                    # On non-sampling frames this is the previous label, redrawn
                    # against the current (freshly detected) box.
                    stable_emotion, conf, face_stressed = face_label
                    if face_stressed:
                        face_alert_count = 1

                    x1, y1, x2, y2 = box
                    color = (0, 0, 255) if face_stressed else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{stable_emotion} ({conf:.2f}) [{angle:.0f} deg]",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            # ---- fuse audio + face into one alert level (same rules as Final) ----
            level, reason = fuse(audio_hot, face_alert_count, n_faces)
            now = time.time()
            if level > 0:
                if level >= held_level or now >= level_until:
                    held_level, held_reason = level, reason
                level_until = now + max(HOLD_SECONDS, 1.0 if level >= 2 else 0.0)
            if now >= level_until:
                held_level, held_reason = 0, ""

            if held_level > 0:
                bcolor, text = LEVEL_STYLE[held_level]
                cv2.circle(frame, (25, 25), 12, bcolor, -1)
                cv2.putText(frame, f"{text} - {held_reason}", (45, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bcolor, 2)

            hud = audio.hud()
            cv2.putText(frame, hud, (20, frame_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

            frame_count += 1
            cv2.imshow("Cabin Disturbance Detection (MediaPipe + HSEmotion + YAMNet)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        audio.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
