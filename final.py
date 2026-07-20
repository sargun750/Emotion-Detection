import cv2
import torch
import time
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

print("Loading model... (first run downloads ~330MB)")
processor = ViTImageProcessor.from_pretrained('abhilash88/face-emotion-detection')
model = ViTForImageClassification.from_pretrained('abhilash88/face-emotion-detection')
model.eval()

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
ALERT_EMOTIONS = {'Angry', 'Disgust', 'Fear', 'Sad'}
NEUTRAL_IDX = emotions.index('Neutral')
NEUTRAL_THRESHOLD = 0.30
ALERT_CONF_THRESHOLD = 0.60
HOLD_SECONDS = 0.3

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)
frame_count = 0
face_labels = {}
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

    run_model = (frame_count % 5 == 0)
    alert_count = 0

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if run_model:
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)

            inputs = processor(pil_img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                preds = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

            idx = torch.argmax(preds).item()
            emotion = emotions[idx]
            conf = preds[idx].item()

            is_alert = (emotion in ALERT_EMOTIONS and conf > ALERT_CONF_THRESHOLD) or (emotion == 'Neutral' and conf < NEUTRAL_THRESHOLD)
            face_labels[i] = (f"{emotion} ({conf:.2f})", is_alert)

        label, is_alert = face_labels.get(i, ("", False))
        if is_alert:
            alert_count += 1

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if run_model:
        face_labels = {k: v for k, v in face_labels.items() if k < len(faces)}

    if run_model:
        if alert_count >= 1:
            alert_until = time.time() + HOLD_SECONDS
        if len(faces) > 1 and alert_count >= 2:
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