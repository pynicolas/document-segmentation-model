# FairScan Segmentation Model

This repository contains a lightweight, custom-trained model for document segmentation.
Its purpose is to provide [FairScan](https://github.com/pynicolas/FairScan) with automatic document detection.

## Overview

- **Model architecture**: DeepLabV3Plus with MobileNet v2 encoder
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
[fairscan-segmentation-dataset](https://github.com/pynicolas/fairscan-segmentation-dataset/).
It's automatically downloaded in the training script.

## Training
```
# 1. Clone the repository
git clone https://github.com/pynicolas/fairscan-segmentation-model
cd fairscan-segmentation-model

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

## Acknowledgements

This project builds on the work of several excellent open-source projects:

- [PyTorch](https://pytorch.org/) — Deep learning framework (BSD-3-Clause)
- [Albumentations](https://github.com/albumentations-team/albumentations) — Image augmentation library (MIT)
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) — High-level segmentation models including U-Net (MIT)
- [OpenCV](https://opencv.org/) — Image processing library (Apache-2.0)
- [scikit-learn](https://scikit-learn.org/) — Machine learning utilities (BSD-3-Clause)
- [ai-edge-torch](https://github.com/google-research/ai-edge) — Model quantization and export to TFLite (Apache-2.0)

We gratefully acknowledge the work of the authors and communities behind these libraries.


