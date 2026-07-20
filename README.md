# Real-Time Emotion & Stress Detection

A real-time facial emotion recognition system that flags stress from a webcam feed, built as a foundation for an in-cabin driver and passenger monitoring application.

## Idea

The system watches a live video feed, detects faces, and classifies each face into one of seven emotions using a Vision Transformer (ViT) model. Certain emotions are treated as **stress indicators**. When a stressed state is detected, a red alert marker appears in the top-left corner of the frame. When multiple people are stressed at once, it escalates to a **high alert**.

The end goal is an in-vehicle monitoring aid: track the emotional state of the driver and passengers so the app can react to trouble scenarios — a distressed driver, a frightened passenger, or an escalating argument in the cabin.

## How It Works (Logic)

The pipeline runs per video frame:

1. **Capture & preprocess** — read a frame from the webcam, mirror it, and resize it.
2. **Face detection** — an OpenCV Haar cascade locates all faces in the frame and crops each one.
3. **Emotion classification** — each cropped face is passed through the ViT model, and a softmax gives a probability for each of the seven emotions. To keep the video smooth on CPU, the model runs every 5th frame and reuses the last labels in between.
4. **Per-face stress decision** — a face is treated as *stressed* if **either**:
   - its top emotion is `Angry`, `Disgust`, `Fear`, or `Sad`, **or**
   - its top emotion is `Neutral` but the confidence is below 60% (an uncertain-neutral is treated as possible hidden stress).

   `Happy`, `Surprise`, and confident `Neutral` produce no alert.
5. **Frame-level alert levels** — after all faces are scored, the system counts how many are stressed:
   - **Alert – Stressed:** at least one face is stressed.
   - **HIGH ALERT – Multiple Stressed:** more than one face is present **and** at least two of them are stressed (e.g. 2 of 3 occupants). High alert takes display priority.
6. **Sticky alerts** — because the model is sampled every few frames and can flicker, alerts are latched for a short hold window so the marker stays steady instead of blinking.

### Why these rules

The model is biased toward predicting `Neutral` and has moderate accuracy on negative emotions. The alert logic is deliberately **sensitive**: since the model under-reports negative emotions, a non-neutral prediction is treated as a strong signal, and a low-confidence neutral is treated as "not clearly calm." The high-alert tier turns individual stress signals into a cabin-level cue for a possible group conflict.

## Model

- **Repository:** `abhilash88/face-emotion-detection` (Hugging Face)
- **Architecture:** Vision Transformer (ViT-Base), fine-tuned from `google/vit-base-patch16-224`
- **Dataset:** FER2013 (~35,887 images, 7 emotion classes)
- **Reported accuracy:** ~71.5% · **Size:** ~86M parameters (~330MB)
- **Execution:** Runs **locally**. Weights download once to a local cache on first run via the `transformers` library, then all inference happens on your own CPU — no API calls.

Emotion classes: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`.

## Dependencies

- Python 3.8+
- `transformers` — loads and runs the ViT model
- `torch` — deep-learning backend (CPU build works; prebuilt wheel, no C++ compiler needed)
- `opencv-python` — webcam capture, face detection, drawing, display
- `pillow` — image conversion between OpenCV and the model

Install:

```bash
pip install transformers torch opencv-python pillow
```

## How to Run

1. Install the dependencies above and connect a webcam.
2. Run the script:

   ```bash
   python emotion_detection.py
   ```

3. On first run the model downloads (~330MB); later runs load from cache and start immediately.
4. A window opens with the live feed. Each face is boxed with its predicted emotion, and an alert / high-alert marker appears in the top-left corner when stress is detected.
5. Press **`q`** to quit.

### Tuning

- `NEUTRAL_THRESHOLD` (default `0.60`) — raise for more sensitive alerts, lower for stricter.
- `HOLD_SECONDS` — how long an alert stays latched; raise for a steadier marker.
- `frame_count % 5` — how often the model runs; lower for fresher labels at the cost of CPU/smoothness.

## Use Case: Driver & Passenger Monitoring

Because OpenCV's detector returns *all* faces in a frame, both the driver and passengers can be monitored at once.

### Trouble scenarios it can help flag

- **Driver stress / road rage** — sustained `Angry` or `Fear` on the driver may indicate aggressive driving or a stressful situation.
- **Distressed or frightened passenger** — a passenger repeatedly showing `Fear` or `Sad` may signal discomfort, illness, or an unsafe situation.
- **Escalation / possible altercation** — when multiple occupants are stressed at once, the **high-alert** tier fires, giving a reasonable heuristic for a heated argument or fight brewing in the cabin.
- **General cabin stress** — a broad rise in negative or uncertain emotions can trigger a check-in or notification.

### Extending toward a real product

The current build is a prototype-grade demo. A serious deployment would add:

- **Role assignment & tracking** — tag which face is the driver and track identities across frames with a lightweight tracker, so roles don't jump between people.
- **Drowsiness & distraction** — eye-closure and head-pose are more safety-critical than emotion for the driver and should be primary signals.
- **Real fight detection** — facial emotion only gives a coarse heuristic; genuine fight detection needs body-pose/motion analysis and/or audio cues (raised voices), which are often the strongest signal in a cramped cabin.
- **Robustness** — variable lighting, motion blur, sunglasses, and partial faces in a moving vehicle are handled poorly by a FER2013-trained model.

## Limitations

- The model is biased toward `Neutral` and is only ~71% accurate.
- Runs on CPU, so inference is sampled every few frames to keep the video responsive.
