"""
One-shot: convert the HSEmotion emotion classifier from ONNX to TFLite.

Source model is the same one `hsemotion_onnx` downloads and caches:
    ~/.hsemotion/enet_b0_8_best_afew.onnx     (EfficientNet-B0, AffectNet, 8 classes)

Output:
    Emotion Detection/Models/hsemotion_enet_b0_8.tflite

Conversion goes ONNX -> TF SavedModel -> TFLite via `onnx2tf`, chosen over the older
`onnx-tf` because it rewrites NCHW->NHWC properly instead of wrapping every conv in
transposes. NOTE the consequence: the ONNX model takes [1,3,224,224] (NCHW) but the
converted TFLite takes [1,224,224,3] (NHWC), so the preprocessing in READY_FOR_TFLITE.py
must NOT do the transpose(2,0,1) that hsemotion_onnx does.

Run once:
    python convert_hsemotion_tflite.py
Then check it with:
    python verify_tflite_parity.py
"""

import contextlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnx2tf


@contextlib.contextmanager
def stub_onnx2tf_calibration():
    """
    onnx2tf runs an ONNX-vs-TF sanity check on any 4D float32 image input, and pulls
    its own calibration .npy to do it. That download is unconditional (no flag turns
    it off) and the copy on this machine is corrupt, so the conversion dies before it
    starts. Substitute a synthetic batch for the duration of the convert call.

    This is an in-memory monkeypatch only - it writes nothing and touches no file
    outside this folder. The check it feeds is a rough smoke test anyway; real
    accuracy is proven by verify_tflite_parity.py against onnxruntime.
    """
    import onnx2tf.onnx2tf as _mod

    original = _mod.download_test_image_data
    rng = np.random.default_rng(0)
    _mod.download_test_image_data = lambda: rng.random(
        (20, 224, 224, 3), dtype=np.float32)
    try:
        yield
    finally:
        _mod.download_test_image_data = original

HERE = Path(__file__).resolve().parent
MODELS = HERE / "Models"
MODELS.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "enet_b0_8_best_afew"
ONNX_CACHE = Path.home() / ".hsemotion" / f"{MODEL_NAME}.onnx"
ONNX_URL = ("https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main"
            f"/models/affectnet_emotions/onnx/{MODEL_NAME}.onnx?raw=true")

OUT_TFLITE = MODELS / "hsemotion_enet_b0_8.tflite"
FOLDED_ONNX = MODELS / "hsemotion_folded.onnx"   # intermediate, kept for inspection


def fold_dynamic_padding(src):
    """
    Make the graph statically shaped before handing it to onnx2tf.

    PyTorch exports EfficientNet's stride-2 "same padding" as five `Pad` nodes whose
    pad amounts are COMPUTED AT RUNTIME from a Shape/Gather/Concat subgraph. onnx2tf
    cannot fold a runtime-valued pad into TFLite's SAME/VALID, so those five convs
    come out as FlexConv2D / FlexDepthwiseConv2dNative - Select-TF ops that need the
    Flex delegate (i.e. full TensorFlow) at runtime, defeating the whole migration.

    Pinning the batch dim to 1 makes every shape static, and onnxruntime's BASIC
    constant folding then evaluates that subgraph away, folding the padding into each
    Conv's own attribute. 533 nodes -> 239, and no Pad nodes remain.

    BASIC specifically: ORT's EXTENDED level also rewrites the graph into
    com.microsoft ops (FusedConv, QuickGelu) that onnx2tf cannot read.
    """
    import onnxruntime as ort

    model = onnx.load(str(src))
    dim0 = model.graph.input[0].type.tensor_type.shape.dim[0]
    dim0.ClearField("dim_param")
    dim0.dim_value = 1

    staged = MODELS / "_static_batch.onnx"
    onnx.save(model, staged)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(FOLDED_ONNX)
    ort.InferenceSession(str(staged), opts, providers=["CPUExecutionProvider"])
    staged.unlink(missing_ok=True)

    folded = onnx.load(str(FOLDED_ONNX))
    pads = sum(1 for n in folded.graph.node if n.op_type == "Pad")
    print(f"[fold] {len(model.graph.node)} -> {len(folded.graph.node)} nodes, "
          f"{pads} Pad nodes remaining")
    if pads:
        print("[fold] WARNING: Pad nodes survived; expect Flex ops downstream.")
    return FOLDED_ONNX


def report_flex(path):
    """Fail loudly if any Select-TF op survived - that would break offline use."""
    from collections import Counter
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(path))
    ops = Counter(d["op_name"] for d in interp._get_ops_details())
    flex = {k: v for k, v in ops.items() if k.startswith("Flex")}
    print(f"[check] {sum(ops.values())} ops, Flex/Select-TF ops: {sum(flex.values())}")
    if flex:
        raise SystemExit(
            f"FAIL: model still contains Select-TF ops {flex}. It would need the Flex "
            f"delegate (full TensorFlow) at runtime, so it is not offline-portable.")
    print("[check] OK - 100% TFLite builtins, no Flex delegate needed.")


def get_onnx():
    """Reuse the already-downloaded cache; only hit the network if it's missing."""
    if ONNX_CACHE.is_file():
        print(f"[onnx] using cached {ONNX_CACHE} ({ONNX_CACHE.stat().st_size/1e6:.1f} MB)")
        return ONNX_CACHE

    ONNX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[onnx] not cached, downloading from {ONNX_URL}")
    urllib.request.urlretrieve(ONNX_URL, ONNX_CACHE)
    print(f"[onnx] saved {ONNX_CACHE} ({ONNX_CACHE.stat().st_size/1e6:.1f} MB)")
    return ONNX_CACHE


def describe_onnx(path):
    m = onnx.load(str(path))
    for i in m.graph.input:
        dims = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"[onnx] input  {i.name} {dims}")
    for o in m.graph.output:
        dims = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"[onnx] output {o.name} {dims}")


def main():
    src = get_onnx()
    describe_onnx(src)
    src = fold_dynamic_padding(src)

    # onnx2tf writes a folder of variants (float32/float16/...) plus a SavedModel.
    # Convert into a temp dir, then keep EVERY .tflite it produced in this folder -
    # the SavedModel clutter is left behind.
    with tempfile.TemporaryDirectory() as tmp:
        print("[convert] running onnx2tf (this takes a minute)...")
        with stub_onnx2tf_calibration():
            onnx2tf.convert(
                input_onnx_file_path=str(src),
                output_folder_path=tmp,
                # The ONNX declares a SYMBOLIC batch dim ('batch_size'). Left symbolic,
                # onnx2tf's shape inference goes wrong and builds a first conv with 224
                # input channels instead of 3. Pinning the full static shape fixes it.
                overwrite_input_shape=["input:1,3,224,224"],
                # Without this, EfficientNet's depthwise/grouped convs convert to
                # tf.nn.convolution with a groups= argument, which has no TFLite
                # builtin and lands as FlexConv2D / FlexDepthwiseConv2dNative. Flex
                # ops need the SELECT_TF_OPS delegate at runtime, i.e. full
                # TensorFlow - exactly the dependency this migration removes.
                # Splitting the group convs keeps the model 100% TFLite builtins.
                disable_group_convolution=True,
                copy_onnx_input_output_names_to_tflite=True,
                output_signaturedefs=True,
                non_verbose=True,
            )

        produced = sorted(Path(tmp).glob("*.tflite"))
        if not produced:
            raise SystemExit(f"onnx2tf produced no .tflite in {tmp}")

        print("[convert] onnx2tf produced, keeping all in this folder:")
        for p in produced:
            dest = MODELS / p.name
            shutil.copy2(p, dest)
            print(f"    {dest.name:50s} {dest.stat().st_size/1e6:6.2f} MB")

        # Canonical name that READY_FOR_TFLITE.py loads.
        float32 = [p for p in produced if "float32" in p.name]
        chosen = float32[0] if float32 else produced[0]
        shutil.copy2(chosen, OUT_TFLITE)
        print(f"[convert] canonical: {chosen.name} -> {OUT_TFLITE.name}")

    report_flex(OUT_TFLITE)

    # Report the real signature - the preprocessing layout depends on it.
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(OUT_TFLITE))
    interp.allocate_tensors()
    for d in interp.get_input_details():
        print(f"[tflite] input  {d['name']} {list(d['shape'])} {d['dtype'].__name__}")
    for d in interp.get_output_details():
        print(f"[tflite] output {d['name']} {list(d['shape'])} {d['dtype'].__name__}")

    print(f"\nDone: {OUT_TFLITE} ({OUT_TFLITE.stat().st_size/1e6:.1f} MB)")
    print("Next: python verify_tflite_parity.py")


if __name__ == "__main__":
    main()
