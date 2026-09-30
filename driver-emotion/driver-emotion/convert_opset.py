import onnx
from onnx import version_converter

model = onnx.load('emotion-ferplus-8.onnx')
model = version_converter.convert_version(model, 12)
onnx.save_model(model, 'emotion-ferplus-12.onnx')

print("Converted. New opset:", model.opset_import[0].version)