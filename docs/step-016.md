# Step 016: CNN/Object-Detection Model Examples

This document describes the complete CNN/detector model examples.

Step 016 adds runnable examples under [../examples/cnn](../examples/cnn):

- ResNet-style residual block,
- FPN-style top-down neck,
- YOLO/SSD-style detection head,
- small full detector on a synthetic image.

Each example builds a unified Model IR graph and has a NumPy golden reference.

---

## 1. Examples

### ResNet block

File: [../examples/cnn/resnet_block.py](../examples/cnn/resnet_block.py)

`Conv2d -> ReLU -> Conv2d -> residual Add`.

### FPN-style neck

File: [../examples/cnn/fpn_neck.py](../examples/cnn/fpn_neck.py)

`lateral Conv2d -> ReLU -> 4D Resize -> Add` with a projected feature map.

### Detection head

File: [../examples/cnn/detector_head.py](../examples/cnn/detector_head.py)

`BoxDecode -> NMS`.

### Full detector

File: [../examples/cnn/full_detector.py](../examples/cnn/full_detector.py)

`Conv2d -> ReLU -> MaxPool -> Reshape` plus `BoxDecode -> NMS`, producing both
feature and detection outputs.

---

## 2. Codegen additions

To support these examples, the codegen now handles:

- elementwise add/sub/mul/div on arbitrary tensor ranks,
- 4D nearest-neighbor resize,
- 4D channel concatenation.

These are in [../compiler/codegen/common_kernels.py](../compiler/codegen/common_kernels.py)
and [../compiler/codegen/shape_kernels.py](../compiler/codegen/shape_kernels.py).

---

## 3. Files changed

- [../examples/cnn/__init__.py](../examples/cnn/__init__.py)
- [../examples/cnn/resnet_block.py](../examples/cnn/resnet_block.py)
- [../examples/cnn/fpn_neck.py](../examples/cnn/fpn_neck.py)
- [../examples/cnn/detector_head.py](../examples/cnn/detector_head.py)
- [../examples/cnn/full_detector.py](../examples/cnn/full_detector.py)
- [../tests/model_e2e/__init__.py](../tests/model_e2e/__init__.py)
- [../tests/model_e2e/test_cnn_models_e2e.py](../tests/model_e2e/test_cnn_models_e2e.py)

---

## 4. Tests

New model-level tests run all four examples through the shared E2E harness and
compare them with NumPy references.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
