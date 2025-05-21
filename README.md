# MICCAI 2025 #1568 Submission

Hierarchically Decoupled Learning with MoEs for Freehand 3D Ultrasound Reconstruction

**The DeNet weights and some testing data are available in [release](https://github.com/MICCAI-2025-1568/DeNet/releases).**

### Setup
- Environment
    ```shell
    git clone https://github.com/MICCAI-2025-1568/DeNet
    cd DeNet
    pip install -r requirements.txt
    ```
- DeNet weights: Download the [denet_Arm.pth](https://github.com/MICCAI-2025-1568/DeNet/releases/download/Weights-and-Data/denet_Arm.pth), then place the file in the folder `weights`.
- Testing data: Download the [data.zip](https://github.com/MICCAI-2025-1568/DeNet/releases/download/Weights-and-Data/data.zip), then extract the file in the folder `cases`.

### Training
```shell
python -m main -m denet -d Arm -r hp -g0
```

### Inference
```shell
python -m infer
```

### Interactive Demo
https://miccai-2025-1568.github.io/MICCAI-2025-1568