# Real-Time Emotion & Stress Detection

A real-time in-cabin disturbance detector. It watches the driver and passengers through a camera, listens to the cabin through a microphone, and raises an alert when it sees a stressed face **and** hears a raised voice at the same time.

The directory holds three working builds, oldest to newest:

| Script | Face pipeline | Emotion model | Audio model | Offline? |
|---|---|---|---|---|
| `Final_audio_visual.py` | Haar cascade + IoU tracker, **multi-face** | ViT, 7 classes (~330 MB) | YAMNet via TF Hub | no — downloads on first run |
| `READY TO GO.py` | MediaPipe Face Landmarker, single face | HSEmotion ONNX, 8 classes | YAMNet via TF Hub | no — downloads on first run |
| **`READY_FOR_TFLITE.py`** | MediaPipe Face Landmarker, single face | HSEmotion **TFLite**, 8 classes | YAMNet **TFLite** | **yes — nothing is downloaded** |

**`READY_FOR_TFLITE.py` is the final script.** The other two are kept because they are the lineage that produced it, and because `Final_audio_visual.py` is still the only build that handles more than one face.

## Idea

The end goal is an in-vehicle monitoring aid: track the emotional state of the driver and passengers so the app can react to trouble scenarios — a distressed driver, a frightened passenger, or an escalating argument in the cabin.

Faces alone are not enough. A frown is not an emergency, and a shout could be someone singing along to the radio. So the system uses **two senses and requires them to agree**: a full red alert fires only when a raised voice and a stressed expression occur together. Either signal alone is shown as a weak, informational cue.

The final build carries a further constraint: it must run **with no network at all**. A car is not a place where you can rely on a model downloading itself, so every weight file lives in the repository and every model runs on TensorFlow Lite.

## How It Works (Logic)

Per video frame, `READY_FOR_TFLITE.py` runs:

1. **Capture & mirror** — read a frame from the webcam and flip it horizontally, so the occupant sees themselves as in a mirror. The flip happens *before* detection, so boxes and labels stay aligned with the image and drawn text reads normally instead of backwards.
2. **Face detection & landmarks** — MediaPipe's Face Landmarker returns 468 facial landmarks in a single pass. Unlike a Haar cascade it does not lose the face when the head tilts.
3. **Alignment** — the outer eye corners (landmarks **33** and **263**) give the head-tilt angle. The frame is rotated so the eyes are level *before* the face is cropped, so the classifier always sees an upright face.
4. **Emotion classification** — the aligned crop is resized to 224×224, ImageNet-normalised, and run through the HSEmotion model, giving a score for each of 8 emotions. To keep the video smooth the classifier runs only every `detection_interval` frames (you are asked for 3–10 at startup) and its last label is reused in between. **The landmarker still runs every frame**, so the box and tilt angle never freeze.
5. **Temporal smoothing** — raw per-frame scores flicker, so `EmotionSmoother` applies two mechanisms: a rolling average, and a **1.5 s minimum hold** before the displayed label may change again. The hold is wall-clock based, not frame based, so it behaves the same on a slow phone as on a desktop.
6. **Per-face stress decision** — a face counts as *stressed* when its smoothed top emotion clears that emotion's bar in `ALERT_THRESHOLDS`. `Happiness`, `Neutral` and `Surprise` never alert.
7. **Audio analysis** — in parallel, on a background thread, YAMNet classifies the last ~1 s of microphone audio (see the audio section below).
8. **Fusion** — the face and audio verdicts are combined by `fuse()` into one of four levels, and the resulting banner is latched for `HOLD_SECONDS` so it doesn't blink.

### Why these rules

Requiring **both** signals is the core design decision. Facial emotion recognition is noisy in a moving vehicle — variable lighting, motion blur, partial faces — and audio alone cannot distinguish an argument from a loud podcast. Demanding agreement between two weak, independent sensors produces far fewer false alarms than trusting either one.

The smoothing exists for the same reason. A single frame of `Anger` means nothing; `Anger` sustained across a full second of averaged evidence means something. The minimum-hold timer additionally prevents the label from oscillating between two similar emotions.

The rolling-average **window is measured in seconds, not frames**. Because the classifier is sampled every `detection_interval` frames, a fixed sample count would silently stretch: 8 samples at interval 10 spans 80 frames, several seconds of lag. Deriving the sample count from `SMOOTHING_SECONDS` keeps the behaviour identical whether you pick interval 3 or 10.

## Model

Three models run in the final build, all as TensorFlow Lite, all stored in `Models/`:

| File | Size | Role |
|---|---|---|
| `face_landmarker.task` | 3.76 MB | MediaPipe face detector + 468-point landmarker (a bundle of three TFLite models) |
| `hsemotion_enet_b0_8.tflite` | 16.05 MB | emotion classifier — NHWC `[1,224,224,3]` → `[1,8]` |
| `yamnet.tflite` | 4.13 MB | audio event classifier — `[15600]` → `[1,521]` |
| `yamnet_class_map.csv` | 15 KB | the 521 AudioSet class names, index-aligned with the model |

**Emotion classifier — HSEmotion `enet_b0_8_best_afew`.** An EfficientNet-B0 backbone trained on **AffectNet**, published by the HSE research group. It was chosen over the FER2013/FERPlus family after testing showed those models were effectively blind to negative emotions — in live tests `anger` and `sadness` never rose above ~9 % confidence during a deliberately angry expression, while `neutral` stayed pinned at 85–92 %. That is a known consequence of severe class imbalance in FER-family datasets, and it makes them useless for stress detection specifically. AffectNet has better coverage of negative emotions, and this model reaches 84 % on anger at peak.

Emotion classes (**this order is the model's output axis — do not reorder**):
`Anger`, `Contempt`, `Disgust`, `Fear`, `Happiness`, `Neutral`, `Sadness`, `Surprise`

Note these differ from the legacy ViT build's 7 classes — `Angry`/`Sad` there are `Anger`/`Sadness` here.

**Legacy model, `Final_audio_visual.py` only:** ViT-Base fine-tuned from `google/vit-base-patch16-224` on FER2013 (`abhilash88/face-emotion-detection`, ~86M params, ~330 MB, ~71.5 % reported accuracy), with classes `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`.

## Dependencies

Python 3.8+ (developed on 3.11). Every package used by any of the three scripts:

| Package | Needed by | Why |
|---|---|---|
| `opencv-python` | all three | webcam capture, drawing, image ops, display |
| `numpy` | all three | array maths |
| `sounddevice` | all three | microphone capture for the audio path |
| `mediapipe` | `READY TO GO.py`, `READY_FOR_TFLITE.py` | Face Landmarker (detection + 468 landmarks) |
| `ai-edge-litert` | `READY_FOR_TFLITE.py` | the TFLite interpreter that runs both models |
| `hsemotion-onnx` | `READY TO GO.py` | emotion classifier via ONNX Runtime |
| `tensorflow` | `Final_audio_visual.py`, `READY TO GO.py` | runs YAMNet |
| `tensorflow_hub` | `Final_audio_visual.py`, `READY TO GO.py` | downloads/loads the YAMNet SavedModel |
| `transformers` | `Final_audio_visual.py` | loads and runs the ViT model |
| `torch` | `Final_audio_visual.py` | backend for the ViT (CPU build is fine) |
| `pillow` | `Final_audio_visual.py` | image conversion between OpenCV and the ViT |
| `onnx` | conversion scripts only | reads/edits the ONNX graph |
| `onnxruntime` | conversion scripts only | constant folding, and the parity reference |
| `onnx2tf` | conversion scripts only | ONNX → TensorFlow → TFLite conversion |

Install for the **final script only** — this is all you need to run it:

```bash
pip install opencv-python numpy sounddevice mediapipe ai-edge-litert
```

Install everything, including the legacy builds and the model-conversion toolchain:

```bash
pip install opencv-python numpy sounddevice mediapipe ai-edge-litert \
            hsemotion-onnx tensorflow tensorflow_hub transformers torch pillow \
            onnx onnxruntime onnx2tf
```

Note `ai-edge-litert` rather than `tf.lite` — on TensorFlow 2.21 `tf.lite.Interpreter` is broken by a signature mismatch against the bundled LiteRT wrapper.

## How to Run

1. Install the dependencies above, connect a webcam, and make sure a microphone is available.
2. Confirm `Models/` contains the four asset files listed under **Model**. They are committed, so a fresh clone already has them (make sure Git LFS is pulled, or they will be pointer files).
3. Run the final script:

   ```bash
   python READY_FOR_TFLITE.py
   ```

4. It asks for a **detection interval** (3 = responsive, 10 = smooth). This is how many frames pass between emotion classifications.
5. A window opens with the mirrored live feed. The face is boxed with its smoothed emotion, confidence and tilt angle; an alert banner appears top-left; a yellow audio HUD runs along the bottom.
6. Press **`q`** to quit.

Nothing is downloaded at any point. The console should show `[audio] loading yamnet.tflite (local, no download)...` followed by `[audio] armed on: [...]` and `listening`.

To run the older builds instead:

```bash
python "READY TO GO.py"        # quotes: the filename contains spaces
python Final_audio_visual.py
```

Both prompt for the same detection interval, and both will download their model weights on first run.

### Tuning

Face-side constants at the top of `READY_FOR_TFLITE.py`:

- `ALERT_THRESHOLDS` — the per-emotion confidence bar for counting a face as stressed. Lower a value to make that emotion alert more readily.
- `SMOOTHING_SECONDS` — how much real time the rolling average spans. Raise for a steadier label, lower for a faster reaction.
- `MIN_HOLD_SECONDS` — minimum time before the displayed label may change again.
- `CONFIDENCE_THRESHOLD` — the averaged score a new emotion must clear before it is allowed to take over.
- `CROP_MARGIN` — padding around the detected face box, as a fraction of its size.
- `HOLD_SECONDS` — how long the fusion banner stays latched.
- the **detection interval** — prompted at startup; lower for fresher labels at the cost of CPU.

## Audio + Visual Variant (`Final_audio_visual.py`)

`Final.py` watches faces only. `Final_audio_visual.py` added the second sense — **hearing** — and the fusion rule that both newer scripts still use. Everything in this section describes the audio path, which is **shared essentially verbatim by all three scripts**; only the model loading differs in `READY_FOR_TFLITE.py`.

### What is YAMNet?

**YAMNet** is a pretrained audio-event classifier published by Google. It was trained on **AudioSet** and recognises **521 sound classes** — including `Speech`, `Shout`, `Screaming`, `Yell`, `Music`, `Singing`, and hundreds more. You feed it a **16 kHz mono waveform** and it returns a **score (0–1) for every one of the 521 classes**. We never train it — it is used off the shelf.

The older scripts fetch it from TensorFlow Hub (`https://tfhub.dev/google/yamnet/1`) as a SavedModel. `READY_FOR_TFLITE.py` uses Google's official prebuilt **TFLite** build instead, stored locally.

### How the audio path works

All audio logic lives in the `AudioAlerter` class, running on a **background thread** so it never stalls the video loop.

1. **Microphone capture** — `sounddevice` opens an input stream at **16 kHz, mono, float32** (`SR`), delivering 100 ms blocks (`BLOCK_SAMPLES`) into a rolling buffer that always holds the last ~0.975 s of audio (`WINDOW_SEC`, YAMNet's native frame length — exactly 15600 samples).
2. **Periodic analysis** — every `ANALYZE_EVERY` (0.5 s) the worker snapshots that ~1 s window and runs YAMNet on it.
3. **Silence gate** — it first computes the window's **RMS** (loudness). If RMS is below `RMS_FLOOR`, the cabin is treated as silent and YAMNet is skipped entirely.
4. **Peak scores** — the code takes the **per-window peak** for each class (`arr.max(axis=0)`), not the mean. A shout is brief, and averaging it across a 1 s window would bury it — the peak keeps short events alive.
5. **Class trigger** — for each name in `ALERT_CLASSES`, if its peak score reaches that class's threshold, the alert **fires**.
6. **Loud-speech fallback** — in practice YAMNet labels *ordinary raised-voice shouting* as `Speech`, not `Shout` (those classes are reserved for extreme, high-pitched sounds). So the alert **also** fires when speech is present (`speech ≥ SPEECH_PRESENT`) **and** the audio is loud (`rms ≥ LOUD_SPEECH_RMS`). This is what actually catches normal shouting; `LOUD_SPEECH_RMS` is the dividing line between "loud talking" and "shouting".
7. **Music veto** — if an alert fired but **music dominates** (`music > MUSIC_VETO` and `music > speech`), the alert is cancelled, so loud music or singing can't masquerade as a disturbance.
8. **Hold** — once fired, the audio alert stays "hot" for `AUDIO_HOLD_SECONDS`, giving the independently sampled face model time to register a matching expression.

### The three class groups

YAMNet scores 521 classes; these three sets pick out the handful that matter:

- **`ALERT_CLASSES`** — the **trigger** list, `name → threshold`. These are the *only* classes that can raise an audio alert. Lower threshold = easier to fire. (`Screaming`/`Shout` are the most sensitive at `0.05`.)
- **`SPEECH_CLASSES`** — a **reference** group, *not* a trigger. Collapsed to one "is a human speaking?" score. Normal talking lands here and deliberately does **not** alert on its own; the score is only used by the loud-speech rule and the music veto.
- **`MUSIC_CLASSES`** — the other reference group, collapsed to a "how musical is this?" score, used solely by the music veto.

### Fusing audio and face (`fuse`)

The final on-screen level combines the audio state with the number of stressed faces. **A full ALERT requires BOTH signals**:

| Signals present | Level | Banner |
|---|---|---|
| Shout **+** ≥2 stressed faces | 3 | red `HIGH ALERT – SHOUT + MULTIPLE STRESSED FACES` |
| Shout **+** ≥1 stressed face | 2 | red `ALERT – SHOUT + STRESSED FACE` |
| Shout only | 1 | orange `WEAK – SHOUT ONLY` |
| Stressed face only | 1 | orange `WEAK – STRESSED FACE ONLY` |
| Neither | 0 | — |

Level 3 is reachable only in `Final_audio_visual.py`. The two newer scripts run the landmarker with `num_faces=1`, so their stressed-face count never exceeds 1 and the multi-face branch is unreachable.

### On-screen HUD and debugging

- A yellow status line at the bottom of the frame shows live `rms`, YAMNet's **top overall class** (`top:`), the best matching **alert class** (`alert`), and a `music-veto` tag when a music suppression happened.
- With `DEBUG_AUDIO = True`, each analysis cycle prints YAMNet's **top-5 classes** to the console (`[audio] rms 0.131 | top5: Speech 0.52, ...`). This is the ground truth for tuning — watch it while talking vs. shouting, then set the thresholds. Turn it off once tuned.

### Face gating (optional CPU saver)

`GATE_FACE_ON_AUDIO`, in `Final_audio_visual.py` only, controls whether the 330 MB ViT runs only while audio is "hot" (to keep it asleep in a quiet cabin). It is currently **`False`**, so the face model runs on its normal every-`detection_interval`-frames schedule regardless of audio. Set it `True` to gate the ViT behind audio interest (the HUD then shows `face idle` while it's asleep). The newer scripts do not have this flag — their emotion model is small enough that gating it is not worth the complexity.

### Graceful degradation

If a required audio package is missing, or the microphone fails to open, the audio path **disables itself and prints why** — the program then runs **face-only** rather than crashing. Check the console at startup for `[audio] armed on: [...]` and `listening` (success) versus `[audio] disabled ...` / `failed to start` (fell back to face-only).

### Extra dependencies and running

All dependencies for all three scripts are in the single table under **Dependencies** above. For this script specifically:

```bash
pip install transformers torch opencv-python pillow tensorflow tensorflow_hub sounddevice
python Final_audio_visual.py        # prompts for detection interval (3–10); press q to quit
```

### Audio tuning knobs

Module-level constants, present in all three scripts:

- `LOUD_SPEECH_RMS` — **the main talk-vs-shout dial.** Raise it if normal loud talking triggers false alarms; lower it if real shouts are missed. Read the printed `rms` values and set it between your talking level and your shouting level.
- `ALERT_CLASSES` — the trigger thresholds for the shout/scream classes.
- `SPEECH_PRESENT` — minimum speech score before loud audio is treated as a raised voice (guards against loud non-voice noise).
- `MUSIC_VETO` — how dominant music must be to suppress an alert.
- `AUDIO_HOLD_SECONDS` — how long an audio alert stays hot (widen if the shout and the stressed face keep just missing each other in time).
- `RMS_FLOOR` — loudness below which the cabin is treated as silent and YAMNet is skipped.

## Offline TFLite Build (`READY_FOR_TFLITE.py`)

`READY_FOR_TFLITE.py` is a twin of `READY TO GO.py`: identical detector, identical thresholds, identical fusion and smoothing. Only model loading and tensor plumbing differ — plus the mirror flip.

### Why the conversion was needed

`READY TO GO.py` could not run without a network, and worse, it could *stop* working on a machine that had already run it:

- **YAMNet** was fetched with `tensorflow_hub.load()`, which caches into `%LOCALAPPDATA%\Temp\tfhub_modules`. That is a **temporary** directory — Windows Disk Cleanup and Storage Sense both delete it. Once cleared, the next run tries to re-download, and offline that is a hard failure.
- **HSEmotion** let the `hsemotion_onnx` package pull its 16 MB `.onnx` from a GitHub raw URL into `~/.hsemotion` on first use, with no checksum and nothing in the repository recording the dependency.
- Neither weight file was committed, so a fresh clone on a machine with no internet could not run at all.

Only MediaPipe's `face_landmarker.task` was already local and already TFLite.

### Converting the emotion model (ONNX → TFLite)

Handled by `convert_hsemotion_tflite.py`, which goes ONNX → TensorFlow → TFLite using `onnx2tf`. Two problems had to be solved.

**Problem 1 — runtime-computed padding produced Flex ops.** PyTorch exports EfficientNet's five stride-2 "same padding" convolutions as `Pad` nodes whose pad amounts are computed **at runtime** from a `Shape`/`Gather`/`Concat` subgraph. TFLite's built-in convolution only understands `SAME` and `VALID`, and `onnx2tf` cannot fold a runtime-valued pad into either — so those five convolutions came out as `FlexConv2D` and `FlexDepthwiseConv2dNative`. Flex ops are Select-TF ops requiring the Flex delegate, i.e. **full TensorFlow at runtime** — precisely the dependency the whole exercise was removing.

The fix is to make the graph statically shaped *before* conversion: pin the batch dimension to 1, then let onnxruntime's constant folding evaluate the padding subgraph away. This collapses **533 nodes to 239** and leaves **zero `Pad` nodes**, so every convolution converts to a TFLite builtin.

Use onnxruntime's **`ORT_ENABLE_BASIC`** level specifically. `ORT_ENABLE_EXTENDED` also folds the pads, but it additionally rewrites the graph into `com.microsoft` ops (`FusedConv`, `QuickGelu`) that `onnx2tf` cannot read. The converter hard-fails if any Flex op survives, so this cannot regress silently.

**Problem 2 — the tensor layout flips.** The ONNX model takes **NCHW** `[1,3,224,224]`; `onnx2tf` emits **NHWC** `[1,224,224,3]`. So `TFLiteEmotionRecognizer.preprocess` must **not** perform the `transpose(2,0,1)` that `hsemotion_onnx` does. It reads the layout off the model rather than assuming it, so it stays correct either way.

### Converting the audio model (YAMNet)

Handled by `fetch_yamnet_tflite.py`. YAMNet is **not** converted locally — it is taken from Google's official prebuilt TFLite build, because YAMNet's STFT/RFFT spectrogram frontend has no TFLite builtin. Converting the SavedModel yourself necessarily produces a Flex-dependent model; publishing a separate TFLite build with a different, TFLite-friendly frontend is exactly why Google ships one.

The 521-class map is copied from the local TF Hub cache when present, so the label indices are guaranteed identical to the ones the older scripts were tuned against.

**Consequence worth knowing:** because the frontend differs, the TFLite YAMNet is **not numerically identical** to the SavedModel. Same weights and the same 521-class index space, but per-class scores differ by roughly 0.1. Verified: silence → `Silence`, a 440 Hz tone → `Sine wave`, white noise → `Static`, and all five alert classes resolve. **`LOUD_SPEECH_RMS` and `ALERT_CLASSES` should therefore be re-checked with `DEBUG_AUDIO = True` rather than carried over blindly.**

### Verifying the conversion

`verify_tflite_parity.py` is the gate. It checks the two models differently, because only one of them *can* be numerically identical:

- **Emotion** — 20 fixed-seed inputs through both onnxruntime and TFLite; requires max absolute difference on the 8 softmax scores below `1e-3` and identical argmax every time. Measured: **1.85e-06, zero mismatches.**
- **YAMNet** — label-space alignment and sanity, not numerics, for the reason above.

If the emotion check ever fails, the cause is the conversion or the NHWC/NCHW preprocessing. **Do not "fix" it by adjusting `ALERT_THRESHOLDS`** — those were tuned against the ONNX model's score distribution.

### Rebuilding the models from scratch

```bash
python convert_hsemotion_tflite.py   # ~/.hsemotion/*.onnx -> Models/hsemotion_enet_b0_8.tflite
python fetch_yamnet_tflite.py        # -> Models/yamnet.tflite + yamnet_class_map.csv
python verify_tflite_parity.py       # both checks must pass
```

The `Models/` folder may also contain build leftovers — `hsemotion_folded.onnx` (the constant-folded intermediate), `hsemotion_folded_float32.tflite` (a duplicate of the canonical file) and `hsemotion_folded_float16.tflite` (an 8 MB half-precision variant, useful if you later target mobile). None are needed at runtime.

## Use Case: Driver & Passenger Monitoring

### Trouble scenarios it can help flag

- **Driver stress / road rage** — sustained `Anger` or `Fear` on the driver, together with a raised voice, may indicate aggressive driving or a confrontation.
- **Distressed or frightened passenger** — a passenger repeatedly showing `Fear` or `Sadness` may signal discomfort, illness, or an unsafe situation.
- **Escalation / possible altercation** — a shout combined with a stressed face is a reasonable heuristic for an argument developing in the cabin.
- **General cabin stress** — a broad rise in negative or uncertain emotions can trigger a check-in or notification.

### Extending toward a real product

The current build is a prototype-grade demo. A serious deployment would add:

- **Multi-face support in the current stack** — `READY_FOR_TFLITE.py` runs `num_faces=1`, so it monitors one occupant. Raising that also means restoring per-face tracking, which this stack dropped when it replaced the Haar + `FaceTracker` design.
- **Role assignment & tracking** — tag which face is the driver and hold that identity across frames, so roles don't jump between people.
- **Drowsiness & distraction** — eye-closure and head-pose are more safety-critical than emotion for the driver and should be primary signals. The 468 landmarks already being computed make this cheap to add.
- **Real fight detection** — facial emotion gives only a coarse heuristic; genuine fight detection needs body-pose/motion analysis alongside the audio cues.
- **Mobile deployment** — the models are now TFLite, which is the format Android needs. Remaining work is the app-side integration and re-tuning the thresholds on real in-car footage.
- **Robustness** — variable lighting, motion blur, sunglasses and partial faces in a moving vehicle remain difficult.

## Limitations

- **One face at a time** in the final build (`num_faces=1`). Only `Final_audio_visual.py` handles multiple occupants, and it uses the weaker ViT model.
- **The level-3 "multiple stressed faces" alert is unreachable** in `READY TO GO.py` and `READY_FOR_TFLITE.py`, for the same reason.
- **Audio thresholds are environment-specific.** `LOUD_SPEECH_RMS` depends on the microphone, its gain and cabin noise, and must be tuned per setup with `DEBUG_AUDIO = True`.
- **The TFLite YAMNet's scores differ from the SavedModel's**, so tuning does not transfer exactly between `READY TO GO.py` and `READY_FOR_TFLITE.py`.
- **Emotion recognition is inherently approximate.** AffectNet training helps considerably on negative emotions compared to FER2013, but subtle classes (`Contempt`, `Disgust`, `Fear`) still carry low confidence.
- **CPU-bound**, so the classifier is sampled every few frames rather than run on every one.
- **Not validated on real in-car footage** — all testing so far has been webcam-based.
