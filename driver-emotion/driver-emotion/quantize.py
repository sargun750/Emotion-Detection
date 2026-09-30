from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input='emotion-ferplus-12.onnx',
    model_output='emotion-ferplus-12-int8.onnx',
    weight_type=QuantType.QUInt8
)

print("Quantization done.")