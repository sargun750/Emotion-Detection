"""
Driver emotion detection - desktop validation script.

Pipeline: webcam -> MediaPipe Face Landmarker (detect + align) ->
HSEmotionONNX (classify) -> temporal smoothing (8-frame window + 1.5s min hold)

Requirements:
    pip install mediapipe opencv-python hsemotion-onnx numpy

You also need the MediaPipe face landmarker model file in the same folder:
    wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -OutFile face_landmarker.task

Run:
    python emotion_test_full.py
Press 'q' to quit.
"""

import time
import urllib.request  # must be imported before hsemotion_onnx - works around a
                        # urllib.request attribute bug in that package on newer Python

from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LANDMARKER_MODEL_PATH = "face_landmarker.task"
EMOTION_MODEL_NAME = "enet_b0_8_best_afew"

LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness",
          "Neutral", "Sadness", "Surprise"]

# eye-corner landmark indices in MediaPipe's 468-point face mesh, used to
# measure head tilt and align the crop before it goes to the emotion model
LEFT_EYE_LANDMARK = 33
RIGHT_EYE_LANDMARK = 263

# temporal smoothing knobs
WINDOW_SIZE = 8              # decision is based on the average of the last N frames
MIN_HOLD_SECONDS = 1.5       # once switched, must hold this long (real time) before switching again
CONFIDENCE_THRESHOLD = 0.35  # averaged score must clear this to be eligible to switch

CROP_MARGIN = 0.4  # extra padding around the face bounding box, as a fraction of its size


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Face alignment
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    smoother = EmotionSmoother(
        LABELS,
        window_size=WINDOW_SIZE,
        min_hold_seconds=MIN_HOLD_SECONDS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        display_frame = frame

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            face_crop, box, angle = align_and_crop(frame, landmarks, frame_w, frame_h)

            if face_crop.size > 0:
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                _raw_emotion, raw_scores = emotion_recognizer.predict_emotions(
                    face_crop_rgb, logits=False
                )

                stable_emotion, avg_scores = smoother.update(raw_scores)

                x1, y1, x2, y2 = box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display_frame,
                    f"{stable_emotion} ({angle:.0f} deg)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                score_str = {
                    label: f"{score*100:.0f}%"
                    for label, score in zip(LABELS, avg_scores)
                }
                print(f"[stable={stable_emotion}] [tilt={angle:.1f} deg] {score_str}")

        cv2.imshow("Driver Emotion Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()