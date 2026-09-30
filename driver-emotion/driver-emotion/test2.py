import cv2
import numpy as np
import urllib.request  # must stay before hsemotion_onnx import
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')

base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5
)
landmarker = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = 33
RIGHT_EYE = 263

def align_and_crop(frame, landmarks, w, h, margin=0.4):
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

    left_eye = pts[LEFT_EYE]
    right_eye = pts[RIGHT_EYE]

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bw, bh = x_max - x_min, y_max - y_min
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    box_w = bw * (1 + margin)
    box_h = bh * (1 + margin)

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))

    x1, y1 = int(cx - box_w / 2), int(cy - box_h / 2)
    x2, y2 = int(cx + box_w / 2), int(cy + box_h / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return rotated[y1:y2, x1:x2], (x1, y1, x2, y2), angle

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    display_frame = frame

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        face_crop, box, angle = align_and_crop(frame, landmarks, w, h)

        if face_crop.size > 0:
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            emotion, scores = fer.predict_emotions(face_rgb, logits=False)
            print(f"[tilt={angle:.1f}°]", emotion,
                  {k: f"{v*100:.0f}%" for k, v in zip(
                      ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise'], scores)})

            x1, y1, x2, y2 = box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{emotion} ({angle:.0f}°)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('MediaPipe Emotion Test', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()