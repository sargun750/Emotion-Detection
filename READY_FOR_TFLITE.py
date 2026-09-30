"""
READY FOR TFLITE — In-cabin disturbance detector, fully offline.

Same detector as `READY TO GO.py`, with every model running on TFLite so nothing
is downloaded at runtime. All thresholds, fusion rules, smoothing and the
detection-interval sampling are unchanged; only model loading and tensor plumbing
differ.

  * AUDIO  — YAMNet shout/scream detection via the official prebuilt `yamnet.tflite`
             instead of `tensorflow_hub.load()` (which cached into %TEMP% and broke
             whenever Windows cleaned it up).
  * FACE   — MediaPipe Face Landmarker -> face alignment -> HSEmotion classifier via
             `hsemotion_enet_b0_8.tflite` instead of the `hsemotion_onnx` package
             (which downloaded its weights from GitHub on first use).

A full red ALERT requires BOTH a shout/scream AND a stressed facial expression.

On startup you are asked for a detection interval (3 = responsive, 10 = smooth).
The landmarker still runs on every frame so the box and tilt angle stay smooth;
only the emotion classifier is sampled, and its last label is reused in between.

Requirements:
    pip install mediapipe opencv-python numpy sounddevice ai-edge-litert

Asset files (all in the ./Models/ folder, all committed - nothing is fetched at runtime):
    face_landmarker.task           MediaPipe face detector + landmarker (TFLite bundle)
    hsemotion_enet_b0_8.tflite     emotion classifier, NHWC [1,224,224,3] -> [1,8]
    yamnet.tflite                  audio event classifier, [15600] -> [1,521]
    yamnet_class_map.csv           the 521 AudioSet class names

Build those with:
    python convert_hsemotion_tflite.py
    python fetch_yamnet_tflite.py
    python verify_tflite_parity.py

If sounddevice is missing or the mic fails, the audio path disables itself and the
program degrades to face-only rather than crashing.

NOTE ON TUNING: the TFLite YAMNet uses a different (TFLite-friendly) spectrogram
frontend than the TF Hub SavedModel, so its per-class scores are not identical.
Re-check LOUD_SPEECH_RMS and ALERT_CLASSES with DEBUG_AUDIO=True before trusting
thresholds carried over from `READY TO GO.py`.

Run:
    python READY_FOR_TFLITE.py
Press 'q' to quit.
"""

import time
import csv
import threading
import collections
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ai_edge_litert.interpreter import Interpreter


HERE = Path(__file__).resolve().parent
MODELS = HERE / "Models"

LANDMARKER_MODEL_PATH = str(MODELS / "face_landmarker.task")
EMOTION_TFLITE = MODELS / "hsemotion_enet_b0_8.tflite"
YAMNET_TFLITE = MODELS / "yamnet.tflite"
YAMNET_CLASS_MAP = MODELS / "yamnet_class_map.csv"



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
LOUD_SPEECH_RMS = 0.14   # speech louder than this = shouting. TUNE from [audio] rms:
                         # set between your normal-talk rms and your shout rms.
SPEECH_PRESENT  = 0.20   # min speech score to treat loud audio as a raised voice (not noise)



# Label order is HSEmotion's own 8-class map - do not reorder, it is the model's
# output axis, not a display preference.
LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness",
          "Neutral", "Sadness", "Surprise"]

# ImageNet normalisation, matching hsemotion_onnx.preprocess exactly.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# eye-corner landmark indices in MediaPipe's 468-point face mesh, used to
# measure head tilt and align the crop before it goes to the emotion model
LEFT_EYE_LANDMARK = 33
RIGHT_EYE_LANDMARK = 263

# temporal smoothing knobs
SMOOTHING_SECONDS = 0.5
MIN_SMOOTHING_SAMPLES = 3    # floor, so the average never collapses to a single frame
MIN_HOLD_SECONDS = 1.5       # once switched, must hold this long (real time) before switching again
CONFIDENCE_THRESHOLD = 0.35  # averaged score must clear this to be eligible to switch

CROP_MARGIN = 0.4  # extra padding around the face bounding box, as a fraction of its size

ALERT_THRESHOLDS = {
    'Anger':    0.45,
    'Contempt': 0.50,
    'Disgust':  0.55,
    'Fear':     0.60,
    'Sadness':  0.45,
}

HOLD_SECONDS = 1



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
        self._in_detail = None
        self._out_detail = None
        self._idx = {}
        self._names = []
        self._stream = None

    def start(self):
        try:
            import sounddevice as sd
        except Exception as e:
            self.status = f"audio off ({type(e).__name__})"
            print(f"[audio] disabled, missing dependency: {e}")
            return

        try:
            if not YAMNET_TFLITE.is_file() or not YAMNET_CLASS_MAP.is_file():
                raise FileNotFoundError(
                    f"missing {YAMNET_TFLITE.name} / {YAMNET_CLASS_MAP.name} - "
                    f"run: python fetch_yamnet_tflite.py")

            print(f"[audio] loading {YAMNET_TFLITE.name} (local, no download)...")
            self._model = Interpreter(model_path=str(YAMNET_TFLITE))
            self._model.allocate_tensors()
            self._in_detail = self._model.get_input_details()[0]
            # Identify the scores tensor by shape - this build also exposes
            # embeddings and a spectrogram, and their order is not guaranteed.
            self._out_detail = next(
                d for d in self._model.get_output_details()
                if list(d["shape"])[-1] == 521)

            expected = int(np.prod(self._in_detail["shape"]))
            if expected != WINDOW_SAMPLES:
                self._model.resize_tensor_input(
                    self._in_detail["index"], [WINDOW_SAMPLES])
                self._model.allocate_tensors()
                self._in_detail = self._model.get_input_details()[0]

            with YAMNET_CLASS_MAP.open(encoding="utf-8") as f:
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

        self._model.set_tensor(self._in_detail["index"],
                               wav.reshape(self._in_detail["shape"]))
        self._model.invoke()
        # Reshaped to 2-D (frames, 521) so every line below is unchanged from the
        # SavedModel version. WINDOW_SAMPLES is exactly one YAMNet frame, so this
        # is a single row either way - the peak/mean distinction is preserved
        # rather than quietly collapsed.
        arr = self._model.get_tensor(self._out_detail["index"]).reshape(1, -1)
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




class TFLiteEmotionRecognizer:
    """
    Same maths as hsemotion_onnx.facial_emotions.HSEmotionRecognizer, reading a
    local .tflite instead of downloading an .onnx to ~/.hsemotion on first use.

    One real difference: the ONNX takes NCHW [1,3,224,224], but onnx2tf emits an
    NHWC [1,224,224,3] input, so the transpose(2,0,1) the original does must NOT
    happen here. The layout is read from the model rather than assumed, and
    verify_tflite_parity.py confirms the scores match the ONNX to ~1e-6.
    """

    def __init__(self, model_path):
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"missing {Path(model_path).name} - "
                f"run: python convert_hsemotion_tflite.py")

        self.interp = Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

        shape = list(self.inp["shape"])
        self.channels_first = shape[1] == 3        # [1,3,H,W] vs [1,H,W,3]
        self.img_size = shape[2] if self.channels_first else shape[1]

    def preprocess(self, face_img_rgb):
        x = cv2.resize(face_img_rgb, (self.img_size, self.img_size))
        x = x.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = x[np.newaxis, ...]
        if self.channels_first:
            x = x.transpose(0, 3, 1, 2)
        return np.ascontiguousarray(x, dtype=np.float32)

    def predict_emotions(self, face_img_rgb, logits=True):
        self.interp.set_tensor(self.inp["index"], self.preprocess(face_img_rgb))
        self.interp.invoke()
        scores = self.interp.get_tensor(self.out["index"])[0]

        if not logits:
            e = np.exp(scores - np.max(scores))
            scores = e / e.sum()
        return LABELS[int(np.argmax(scores))], scores



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



def main():
    print("Loading models (all local TFLite, nothing downloaded)...")
    emotion_recognizer = TFLiteEmotionRecognizer(EMOTION_TFLITE)

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

            # Mirror the frame, so the person sees themselves
            # the way a mirror shows them. Flipping BEFORE detection (not
            # just before imshow) keeps the boxes and labels aligned with the image
            # and leaves the drawn text readable rather than reversed.
            frame = cv2.flip(frame, 1)

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
