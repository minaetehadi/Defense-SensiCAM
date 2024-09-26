# Defense-SensiCAM
Enhancing Adversarial Robustness in Object Classification using Edge Information and Sensi-CAM Heatmaps
## Requirements
- Python >= 3.x
- Tensorflow => 1.12.0
- Pytorch >= 1.8
- Numpy >= 1.15.4
- Opencv >= 3.4.2
## Structure
- `AT-Imagenet.py`: Implements adversarial training for the ImageNet dataset.
- `Boundingbox-Generation.py`: Generates bounding boxes based on the regions highlighted by Sensi-CAM.
- `Discrepency-Sample.py`: Calculates the discrepancy between Sensi-CAM heatmaps of clean and adversarial examples.
- `GradCAM-vs-SensiCAM.py`: Compares Grad-CAM and Sensi-CAM to demonstrate the superiority of Sensi-CAM in highlighting critical image regions.
- `model.py`:  AlexNet, VGG19
- `train.py`: Implements the training.
- `validation.py`: Script to validate the model, test adversarial robustness, and visualize Sensi-CAM heatmaps.
- `README.md`: Explaining the repository structure and usage.
## Dataset
Download the dataset.
- [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
- [ImageNet](https://image-net.org/download)

## Pretrained models
Download the pretrained models for imagenet and Cifar10.
- [Cifar10](https://github.com/MadryLab/cifar10_challenge/tree/master)
- [Imagenet](https://github.com/MadryLab/robustness/tree/master/robustness/imagenet_models)

## Attacks
- [FoolBox](https://github.com/bethgelab/foolbox)
 ## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/minaetehadi/Defense-SensiCAM.git
    cd Defense-SensiCAM
    ```
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

    
