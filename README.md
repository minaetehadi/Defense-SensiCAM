# Defense-SensiCAM
Enhancing Adversarial Robustness in Object Classification using Edge Information and Sensi-CAM Heatmaps
## Requirements
- Python >= 3.x
- Tensorflow => 1.12.0
- Pytorch >= 1.8
- Numpy >= 1.15.4
- Opencv >= 3.4.2
## Structure
- **sensicam.py**: Code to generate SensiCAM heatmaps from teacher (pre-trained model) and student models.
- **ospa_loss.py**: OSPA metric-based discrepancy loss function.
- **model_train.py**: Main script for training the student model, including adversarial attack handling.
- **pgd_attack.py**: Code for generating adversarial examples using PGD attack.
- **utils.py**: Utility functions for edge detection, image preprocessing, etc.
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

    
