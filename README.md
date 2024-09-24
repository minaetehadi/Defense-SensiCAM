# Defense-SensiCAM
Enhancing Adversarial Robustness in Object Classification using Edge Information and Sensi-CAM Heatmaps
## Requirements
- Python >= 3.x
- Tensorflow => 1.12.0
- Numpy >= 1.15.4
- Opencv >= 3.4.2
## Structure
- **sensicam.py**: Code to generate SensiCAM heatmaps from teacher and student models.
- **ospa_loss.py**: OSPA metric-based discrepancy loss function.
- **model_train.py**: Main script for training the student model, including adversarial attack handling.
- **pgd_attack.py**: Code for generating adversarial examples using PGD attack.
- **utils.py**: Utility functions for edge detection, image preprocessing, etc.

 ## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/minaetehadi/Defense-SensiCAM.git
    cd Defense-SensiCAM
    ```

    
