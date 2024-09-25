import torch.nn as nn
import torchvision.models as models

class AlexNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetModel, self).__init__()
        self.model = models.alexnet(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

class VGG19Model(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG19Model, self).__init__()
        self.model = models.vgg19(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

# Function to load a model
def load_model(model_name='VGG19', num_classes=10):
    if model_name == 'AlexNet':
        model = AlexNetModel(num_classes=num_classes)
    elif model_name == 'VGG19':
        model = VGG19Model(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model
