import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AlexNetEdgeModel(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetEdgeModel, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)
        
        self.edge_conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),  # For processing Sobel and Canny edges
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(4096 + 128*6*6, num_classes)  # Combining AlexNet's fc features and edge map features

    def forward(self, x_rgb, x_edges):
        # Forward pass through AlexNet for RGB input
        x_rgb = self.alexnet.features(x_rgb)
        x_rgb = x_rgb.view(x_rgb.size(0), -1)  # Flatten the output for the classifier
        
        x_edges = self.edge_conv(x_edges)
        x_edges = x_edges.view(x_edges.size(0), -1)  # Flatten the edge features
        
        # Combine RGB features and edge map features
        x_combined = torch.cat((x_rgb, x_edges), dim=1)

        output = self.fc(x_combined)
        return output

class VGG19EdgeModel(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG19EdgeModel, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        self.vgg19.classifier[6] = nn.Linear(4096, num_classes)

        self.edge_conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),  # For Sobel and Canny edges
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(4096 + 128*7*7, num_classes)  # Combining VGG19's fc features and edge map features

    def forward(self, x_rgb, x_edges):

        x_rgb = self.vgg19.features(x_rgb)
        x_rgb = x_rgb.view(x_rgb.size(0), -1)  # Flatten the output for the classifier

        x_edges = self.edge_conv(x_edges)
        x_edges = x_edges.view(x_edges.size(0), -1)  # Flatten the edge features

        # Combine RGB features and edge map features
        x_combined = torch.cat((x_rgb, x_edges), dim=1)

        # Final fully connected layer
        output = self.fc(x_combined)
        return output
