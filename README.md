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
- Compatible with LiteRT
- Easily reproducible training: run training with a single command

## Dataset

The dataset can be found in a separate repository:
[document-segmentation-dataset](https://github.com/pynicolas/document-segmentation-dataset/).
It's automatically downloaded in the training script.

## Training
```
# 1. Clone the repository
git clone https://github.com/pynicolas/document-segmentation-model
cd document-segmentation-model

# 2. Create a venv
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate.bat on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the training script
python train.py
```

## Requirements

See [requirements.txt](requirements.txt) for details.

## License

This repository is released under the GNU GPLv3 license.
See [LICENSE](LICENSE) for details.

## Goals & Philosophy

This project is part of a broader effort to provide ethical alternatives in mobile software:

 - 100% on-device inference (no data sent to external servers)
 - Fully open source model and training pipeline
 - No user tracking, no advertising

