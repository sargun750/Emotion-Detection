"""
One-shot: get YAMNet as a TFLite model plus its 521-class label map.

Outputs (both into the Models/ folder):
    yamnet.tflite            official prebuilt TFLite classification build (~4 MB)
    yamnet_class_map.csv     521 AudioSet class names, index-aligned with the model

Google publishes an official TFLite build of YAMNet, so we take that rather than
converting the SavedModel ourselves - YAMNet's STFT frontend would otherwise need
SELECT_TF_OPS (Flex), which drags in full TensorFlow at runtime. If the download
fails, the script falls back to converting the locally cached TF Hub SavedModel
with Flex enabled, so it can still finish with no network.

The class map is copied from the local TF Hub cache when present, so the label
indices are guaranteed to be the exact ones the current build already alerts on.

Run once:
    python fetch_yamnet_tflite.py
"""

import csv
import os
import shutil
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "Models"
MODELS.mkdir(parents=True, exist_ok=True)

OUT_TFLITE = MODELS / "yamnet.tflite"
OUT_CLASSMAP = MODELS / "yamnet_class_map.csv"

# Official prebuilt TFLite. The storage.googleapis.com path is the CDN that
# tfhub.dev itself redirects to for ?lite-format=tflite.
TFLITE_URLS = [
    "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/yamnet/classification/tflite/1.tflite",
    "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite",
]

EXPECTED_CLASSES = 521
EXPECTED_SAMPLES = 15600  # 0.975s @ 16kHz - one YAMNet frame


def find_hub_cache():
    """Locate the SavedModel TF Hub already cached, if it survived Temp cleanup."""
    base = Path(os.environ.get("TFHUB_CACHE_DIR")
                or Path(os.environ.get("TEMP", "/tmp")) / "tfhub_modules")
    if not base.is_dir():
        return None
    for d in base.iterdir():
        if (d / "saved_model.pb").is_file():
            return d
    return None


def get_class_map():
    if OUT_CLASSMAP.is_file():
        print(f"[classmap] already present: {OUT_CLASSMAP.name}")
        return

    cache = find_hub_cache()
    src = cache / "assets" / "yamnet_class_map.csv" if cache else None
    if src and src.is_file():
        shutil.copy2(src, OUT_CLASSMAP)
        print(f"[classmap] copied from TF Hub cache: {src}")
    else:
        url = ("https://raw.githubusercontent.com/tensorflow/models/master/"
               "research/audioset/yamnet/yamnet_class_map.csv")
        print(f"[classmap] cache miss, downloading {url}")
        urllib.request.urlretrieve(url, OUT_CLASSMAP)

    rows = list(csv.DictReader(OUT_CLASSMAP.open(encoding="utf-8")))
    if len(rows) != EXPECTED_CLASSES:
        raise SystemExit(f"class map has {len(rows)} rows, expected {EXPECTED_CLASSES}")
    print(f"[classmap] {len(rows)} classes, e.g. 0={rows[0]['display_name']!r} "
          f"6={rows[6]['display_name']!r} 11={rows[11]['display_name']!r}")


def download_tflite():
    for url in TFLITE_URLS:
        try:
            print(f"[tflite] downloading {url}")
            urllib.request.urlretrieve(url, OUT_TFLITE)
            size = OUT_TFLITE.stat().st_size
            if size < 1_000_000:
                raise ValueError(f"suspiciously small ({size} bytes)")
            print(f"[tflite] got {OUT_TFLITE.name} ({size/1e6:.2f} MB)")
            return True
        except Exception as e:
            print(f"[tflite] failed: {e}")
            OUT_TFLITE.unlink(missing_ok=True)
    return False


def convert_from_cache():
    """Offline fallback: convert the cached SavedModel, Flex ops and all."""
    cache = find_hub_cache()
    if cache is None:
        raise SystemExit(
            "Could not download the official TFLite and no TF Hub cache to convert.\n"
            "Connect once, or place yamnet.tflite in the Models/ folder manually.")

    print(f"[tflite] falling back to converting cached SavedModel: {cache}")
    print("[tflite] NOTE: this build needs SELECT_TF_OPS (Flex) => full TensorFlow.")
    import tensorflow as tf

    conv = tf.lite.TFLiteConverter.from_saved_model(str(cache))
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    OUT_TFLITE.write_bytes(conv.convert())
    print(f"[tflite] wrote {OUT_TFLITE.name} ({OUT_TFLITE.stat().st_size/1e6:.2f} MB)")


def describe():
    """Report the real signature. Which output is `scores` must be read, not assumed."""
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(OUT_TFLITE))
    interp.allocate_tensors()

    print("\n[signature]")
    for d in interp.get_input_details():
        print(f"  input  [{d['index']}] {d['name']:30s} {list(d['shape'])} {d['dtype'].__name__}")

    scores_idx = None
    for d in interp.get_output_details():
        shape = list(d["shape"])
        tag = ""
        if shape and shape[-1] == EXPECTED_CLASSES:
            scores_idx = d["index"]
            tag = "  <-- SCORES"
        print(f"  output [{d['index']}] {d['name']:30s} {shape} {d['dtype'].__name__}{tag}")

    if scores_idx is None:
        raise SystemExit(f"No output with last dim {EXPECTED_CLASSES}; cannot locate scores.")

    inp = interp.get_input_details()[0]
    n = int(inp["shape"][-1])
    if n != EXPECTED_SAMPLES:
        print(f"  NOTE: input length {n} != {EXPECTED_SAMPLES}; "
              f"READY_FOR_TFLITE.py resizes the tensor to match its window.")
    print(f"\nscores output index = {scores_idx}")


def main():
    get_class_map()
    if not OUT_TFLITE.is_file():
        if not download_tflite():
            convert_from_cache()
    else:
        print(f"[tflite] already present: {OUT_TFLITE.name} "
              f"({OUT_TFLITE.stat().st_size/1e6:.2f} MB)")
    describe()
    print("\nNext: python verify_tflite_parity.py")


if __name__ == "__main__":
    main()
