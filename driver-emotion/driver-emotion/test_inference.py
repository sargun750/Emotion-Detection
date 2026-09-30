import cv2
import numpy as np
import onnxruntime as ort

# load model
session = ort.InferenceSession('emotion-ferplus-12-int8.onnx')
input_name = session.get_inputs()[0].name

emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness',
                   'anger', 'disgust', 'fear', 'contempt']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))
        input_data = face_resized.astype(np.float32).reshape(1, 1, 64, 64)

        output = session.run(None, {input_name: input_data})[0][0]
        probs = softmax(output)
        print({emotion_labels[i]: f"{probs[i]*100:.0f}%" for i in range(8)})
        top_idx = int(np.argmax(probs))

        label = f"{emotion_labels[top_idx]} ({probs[top_idx]*100:.0f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    cv2.imshow('Emotion Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()