# Driver Emotion Detection — Model Evaluation Log

This project's goal: detect a driver's emotional state (anger, stress, happiness,
etc.) from the phone's front-facing camera, as part of the larger driver
monitoring system. This README documents each model tried, in order, why it
was replaced, and what script tests it.

---

## 1. emotion-ferplus (ONNX Model Zoo) — `test_interface.py`

**Source:** [ONNX Model Zoo](https://github.com/onnx/models) —
`validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx`

**What it is:** A small VGG13-style CNN trained on FERPlus (a relabeled,
crowd-corrected version of the FER2013 dataset). Chosen first because it's
already in ONNX format — no conversion needed — and is tiny (64x64 grayscale
input), making it an obvious first candidate for a mobile pipeline.

**Pipeline used:**
- Face detection: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)
- Preprocessing: crop → resize to 64x64 → grayscale
- Post-processing: softmax over raw output → 8-class scores
  (neutral, happiness, surprise, sadness, anger, disgust, fear, contempt)

**Steps taken:**
- Downloaded the model, converted opset 8 → 12 (`onnx.version_converter`)
  for better quantization support
- Ran `quantize_dynamic` (int8) — reduced size from ~35MB to ~21MB
  (less than the typical ~4x reduction, because dynamic quantization only
  touches `MatMul`/`Gemm` layers, and this model is mostly `Conv` layers —
  full 4x reduction would need static quantization with a calibration set)

**Result — rejected:** Worked fine for `happiness` and `neutral`, but was
essentially blind to negative emotions. In live testing, `anger` and `sadness`
never rose above ~1-9% confidence even during a deliberately angry/sad
expression, while `neutral` stayed pinned at 85-92%. This lines up with a
known issue in FER-family datasets: severe class imbalance (far more
neutral/happy training examples than anger/disgust/fear), which biases the
model toward predicting the majority classes. Since detecting driver stress/
anger is the actual point of this feature, this made the model unusable
as-is.

---

## 2. HSEmotionONNX — `test1.py`

**Source:** [HSEmotionONNX](https://github.com/av-savchenko/hsemotion-onnx)
(Savchenko/HSE research group), model `enet_b0_8_best_afew`

**What it is:** An EfficientNet-B0 backbone trained on AffectNet instead of
FER2013/FERPlus. AffectNet has better coverage and balance for negative
emotions, and this specific model has been benchmarked by its authors on real
Android hardware (Snapdragon 888), which made it a stronger mobile candidate
than a random alternative.

**Pipeline used:**
- Face detection: OpenCV Haar Cascade (same as before)
- Preprocessing: crop → RGB (this model expects color, unlike
  emotion-ferplus's grayscale input)
- Inference via the `hsemotion_onnx` package's `HSEmotionRecognizer` class

**Known issue hit:** `hsemotion_onnx` internally calls
`urllib.request.urlretrieve(...)` but only does `import urllib`, which fails
on newer Python (3.13) with `AttributeError: module 'urllib' has no attribute
'request'`. Fixed by adding `import urllib.request` as the very first import
in the script, before importing `hsemotion_onnx`.

**Result — promising, but two problems found:**
1. **Much better on negative emotions** than emotion-ferplus — anger reached
   84% confidence at peak and held 40-60% across many consecutive frames
   during a genuinely angry expression. Happiness reached 97-99% cleanly.
   Sadness/fear/disgust/surprise worked but with lower confidence — expected,
   since these are inherently harder/subtler classes with less training data
   even in AffectNet.
2. **Failed on head tilt** — when the head was tilted, predictions stopped
   working entirely. Root cause: the Haar Cascade face detector was failing
   to detect a face at all once tilted more than ~15-20°, so no crop ever
   reached the emotion model (not a weakness in the emotion model itself).

---

## 3. Tilt fix attempt — `test2.py`

Two approaches were tried here, in order, to fix the head-tilt problem from
step 2:

**Attempt A — rotation-search with Haar Cascade (rejected):** Rotated the
frame at multiple angles (0°, ±15°, ±30°, ±45°) and re-ran Haar Cascade
detection at each, using whichever angle succeeded. This technically worked as
a concept, but was rejected because: (a) it's slow — up to 7 detection passes
per frame, unsuitable for a phone CPU running continuously; (b) accuracy
actually *dropped*, because running Haar Cascade repeatedly at different
rotations increased the odds of a false-positive "face" match, feeding
garbage crops into the emotion model.

**Attempt B — MediaPipe Face Landmarker (adopted):** Switched to MediaPipe's
Tasks API (`FaceLandmarker`), which handles head rotation natively in a
single detection pass and returns 468 facial landmarks. Used the outer
eye-corner landmarks (indices 33 and 263) to measure the actual head-tilt
angle geometrically and rotate the crop so the eyes are level before it's fed
to the emotion model — real alignment based on measured geometry, not a
guessed angle.

**Known issue hit:** `mp.solutions.face_mesh` (MediaPipe's older/legacy
namespace) raised `AttributeError: module 'mediapipe' has no attribute
'solutions'` — a known breakage in recent MediaPipe releases as that legacy
API is phased out. Fixed by switching entirely to the newer Tasks API
(`mediapipe.tasks.python.vision.FaceLandmarker`), which is also the
actively-maintained API going forward and the one MediaPipe's mobile SDKs use.

**Result — adopted:** Fixed the tilt problem correctly — face detection and
alignment now hold up across a real range of head tilt, and accuracy no
longer degrades the way it did with Haar Cascade.

Model file needed: `face_landmarker.task`, downloaded from Google's MediaPipe
model repo:
```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

---

## 4. Final version with temporal smoothing — `emotion_full.py`

Combines everything above (MediaPipe alignment + HSEmotionONNX
classification) and adds a smoothing layer to stop the displayed emotion from
flickering frame-to-frame, and to reduce false positives from single noisy
frames.

**Smoothing design — two mechanisms:**
1. **8-frame rolling average** — the decision each frame is based on the
   mean of the last 8 raw score vectors, not a single frame's output.
2. **1.5-second minimum hold** — once the displayed emotion switches, it
   cannot switch again until 1.5 real seconds have passed. This is
   time-based (using a wall-clock check), not frame-count-based, so the hold
   duration stays consistent even if frame rate varies (e.g. a phone running
   slower than a desktop).

**Validation:** Tested with a neutral → smile → neutral sequence. The
smoothed output held `Neutral` steady through natural noise, switched
cleanly to `Happiness` once the smile was clearly established, held it
through 100+ frames without flicker during the sustained smile, and
transitioned back to `Neutral` only once the smile had genuinely faded and
new evidence had built up — no premature switching in either direction.

---

## Summary — model comparison

| Model | Format | Negative emotion handling | Tilt robustness | Mobile fit | Outcome |
|---|---|---|---|---|---|
| emotion-ferplus | ONNX (native) | Poor — collapses to Neutral | N/A (not tested, rejected earlier) | Excellent (tiny, 64x64 gray) | Rejected — unusable for anger/stress detection |
| HSEmotionONNX (enet_b0_8_best_afew) | ONNX (native) | Good — real signal on anger, weaker on rarer classes | Broke with Haar Cascade, fixed with MediaPipe | Good — benchmarked on real Android hardware by its authors | **Adopted** |

---

## Open items / not yet done

- Static quantization of the chosen model for mobile (current pipeline is
  desktop-only; deferred earlier in favor of getting a working pipeline
  first)
- Fine-tuning on driver-specific data (e.g. DAiSEE) if in-car footage proves
  harder than webcam testing
- Integration into the Android/iOS ONNX Runtime Mobile pipeline
- Body-language/pose stream (separate from this facial-expression track)
