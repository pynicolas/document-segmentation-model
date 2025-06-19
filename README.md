# Document Segmentation Model

This repository contains a lightweight, custom-trained model for document segmentation, designed for mobile applications.
It is optimized for conversion to TFLite and intended for integration into ethical, privacy-respecting document scanning apps.

## Overview

- **Model architecture**: U-Net with EfficientNet-lite encoder
- **Format**: PyTorch (.pt) and quantized TFLite (.tflite)
- **Use case**: Detecting the area of a document in a photo, as a first step in document scanning workflows
- **Target platform**: Android (LiteRT / TensorFlow Lite)

## Features

- Small model size for fast inference on mobile
- High segmentation accuracy (Dice score > 0.94 on validation set)
- Supports quantization for optimized deployment
- Compatible with LiteRT and on-device ML pipelines

## Repository structure

## Dataset

The dataset consists of annotated document images (segmentation masks).

## Training

## Requirements

See requirements.txt for details.

## Conversion to TFLite

Model conversion uses ai-edge-torch:
```
import ai_edge_torch
from ai_edge_torch.generative.quantize import quant_recipes

model.eval()
sample_args = (torch.randn(1, 3, 256, 256),)
quant_config = quant_recipes.full_int8_dynamic_recipe()
edge_model = ai_edge_torch.convert(model, sample_args, quant_config=quant_config)
edge_model.export("exports/document_model.tflite")
```

## License

This repository is released under the GNU GPLv3 license.
See [LICENSE](LICENSE) for details.

## Goals & Philosophy

This project is part of a broader effort to provide ethical alternatives in mobile software:

 - 100% on-device inference (no data sent to external servers)
 - Fully open source model and training pipeline
 - No user tracking, no advertising

