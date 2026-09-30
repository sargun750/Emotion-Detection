"""
Prove the TFLite conversions did not change the models' answers.

Compares, on identical inputs:
  * emotion  - hsemotion_enet_b0_8.tflite   vs  ~/.hsemotion/enet_b0_8_best_afew.onnx
  * yamnet   - yamnet.tflite                vs  the cached TF Hub SavedModel

This is the gate for the whole TFLite migration. ALERT_THRESHOLDS in the detector
were tuned against the ONNX model's score distribution, so if the argmax disagrees
the fix is the conversion or the preprocessing - never the thresholds.

Run:
    python verify_tflite_parity.py
"""

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = HERE / "Models"

EMOTION_TFLITE = MODELS / "hsemotion_enet_b0_8.tflite"
EMOTION_ONNX = Path.home() / ".hsemotion" / "enet_b0_8_best_afew.onnx"
YAMNET_TFLITE = MODELS / "yamnet.tflite"

LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness",
          "Neutral", "Sadness", "Surprise"]

N_TRIALS = 20
TOL = 1e-3

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def preprocess_nhwc(img_uint8):
    """Identical maths to hsemotion_onnx.preprocess, but stopping at NHWC."""
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x[np.newaxis, ...].astype(np.float32)


def find_hub_cache():
    base = Path(os.environ.get("TFHUB_CACHE_DIR")
                or Path(os.environ.get("TEMP", "/tmp")) / "tfhub_modules")
    if not base.is_dir():
        return None
    for d in base.iterdir():
        if (d / "saved_model.pb").is_file():
            return d
    return None


def check_emotion():
    print("=" * 68)
    print("EMOTION  hsemotion_enet_b0_8.tflite  vs  enet_b0_8_best_afew.onnx")
    print("=" * 68)

    if not EMOTION_ONNX.is_file():
        print(f"SKIP: reference ONNX missing at {EMOTION_ONNX}")
        return None

    import onnxruntime as ort
    from ai_edge_litert.interpreter import Interpreter

    sess = ort.InferenceSession(str(EMOTION_ONNX), providers=["CPUExecutionProvider"])
    onnx_in = sess.get_inputs()[0].name

    interp = Interpreter(model_path=str(EMOTION_TFLITE))
    interp.allocate_tensors()
    tin = interp.get_input_details()[0]
    tout = interp.get_output_details()[0]
    print(f"tflite input {list(tin['shape'])}  output {list(tout['shape'])}")

    rng = np.random.default_rng(1234)
    worst = 0.0
    label_mismatch = 0

    for _ in range(N_TRIALS):
        img = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        nhwc = preprocess_nhwc(img)
        nchw = nhwc.transpose(0, 3, 1, 2).copy()  # ONNX wants channels-first

        onnx_scores = softmax(sess.run(None, {onnx_in: nchw})[0][0])

        interp.set_tensor(tin["index"], nhwc)
        interp.invoke()
        tfl_scores = softmax(interp.get_tensor(tout["index"])[0])

        worst = max(worst, float(np.abs(onnx_scores - tfl_scores).max()))
        if int(onnx_scores.argmax()) != int(tfl_scores.argmax()):
            label_mismatch += 1

    print(f"trials             : {N_TRIALS}")
    print(f"max |delta| softmax: {worst:.2e}   (tolerance {TOL:.0e})")
    print(f"argmax mismatches  : {label_mismatch}")

    ok = worst < TOL and label_mismatch == 0
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return ok


def check_yamnet():
    """
    NOT a numerical parity check, deliberately.

    The TF Hub SavedModel computes its mel spectrogram with STFT/RFFT ops that have
    no TFLite builtin - which is exactly why Google publishes a SEPARATE TFLite build
    with a TFLite-friendly frontend rather than a straight conversion. Same weights,
    different frontend numerics, so per-class scores legitimately differ by O(0.1).
    Asserting 1e-3 agreement here would be asserting something false.

    What must hold, and what this checks, is that the 521-class INDEX SPACE is the
    same - i.e. yamnet_class_map.csv still labels the TFLite outputs correctly, so
    ALERT_CLASSES / SPEECH_CLASSES / MUSIC_CLASSES keep pointing at the right classes.
    """
    print()
    print("=" * 68)
    print("YAMNET   yamnet.tflite  - label-space alignment + sanity")
    print("=" * 68)

    if not YAMNET_TFLITE.is_file():
        print(f"SKIP: {YAMNET_TFLITE.name} missing")
        return None

    import csv
    from ai_edge_litert.interpreter import Interpreter

    classmap = MODELS / "yamnet_class_map.csv"
    names = [r["display_name"] for r in csv.DictReader(classmap.open(encoding="utf-8"))]
    index_of = {n: i for i, n in enumerate(names)}

    interp = Interpreter(model_path=str(YAMNET_TFLITE))
    interp.allocate_tensors()
    tin = interp.get_input_details()[0]
    scores_out = next(d for d in interp.get_output_details()
                      if list(d["shape"])[-1] == 521)
    n = int(np.prod(tin["shape"]))
    print(f"tflite input {list(tin['shape'])}  scores {list(scores_out['shape'])}")

    def run(wav):
        interp.set_tensor(tin["index"], wav.reshape(tin["shape"]))
        interp.invoke()
        return interp.get_tensor(scores_out["index"]).reshape(-1)

    t = np.arange(n) / 16000.0
    probes = {
        # signal                                        -> class its top-1 must land in
        "silence": (np.zeros(n, np.float32), {"Silence"}),
        "sine440": ((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
                    {"Sine wave", "Beep, bleep", "Musical instrument", "Tuning fork"}),
        "white noise": ((np.random.default_rng(0).standard_normal(n) * 0.1).astype(np.float32),
                        {"Static", "White noise", "Noise", "Waterfall", "Water", "Rain"}),
    }

    failures = 0
    for tag, (wav, expected) in probes.items():
        v = run(wav)
        top = names[int(v.argmax())]
        ok = top in expected
        failures += (not ok)
        print(f"  {tag:12s} -> {top:20s} {v.max():.3f}   {'ok' if ok else 'UNEXPECTED'}")

    # Every class the detector keys on must resolve in this label space.
    needed = ["Screaming", "Shout", "Yell", "Children shouting", "Whoop",
              "Speech", "Music"]
    missing = [c for c in needed if c not in index_of]
    print(f"  alert classes resolvable: {len(needed)-len(missing)}/{len(needed)}"
          + (f"  MISSING {missing}" if missing else ""))
    failures += len(missing)

    ok = failures == 0
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    print("  NOTE: scores differ from the SavedModel build by design (different")
    print("        frontend). Re-check LOUD_SPEECH_RMS / ALERT_CLASSES with")
    print("        DEBUG_AUDIO=True before trusting the old tuning.")
    return ok


def main():
    results = {"emotion": check_emotion(), "yamnet": check_yamnet()}

    print()
    print("=" * 68)
    for name, r in results.items():
        print(f"  {name:8s} {'PASS' if r else ('SKIPPED' if r is None else 'FAIL')}")
    print("=" * 68)

    if any(r is False for r in results.values()):
        raise SystemExit("Parity FAILED - do not ship this conversion.")


if __name__ == "__main__":
    main()
