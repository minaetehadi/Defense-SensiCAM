import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2
import numpy as np

# Define AlexNet and VGG19 Models
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

# PGD Attack for Adversarial Example Generation
class LinfPGDAttack:
    def __init__(self, model, epsilon=0.03, num_steps=40, step_size=0.01, random_start=True):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start

    def perturb(self, images, labels):
        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.zeros_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            adv_images = torch.clamp(adv_images, 0, 1)

        adv_images.requires_grad = True

        for _ in range(self.num_steps):
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            adv_images = adv_images + self.step_size * adv_images.grad.sign()
            adv_images = torch.clamp(adv_images, images - self.epsilon, images + self.epsilon)
            adv_images = torch.clamp(adv_images, 0, 1).detach_()
            adv_images.requires_grad = True

        return adv_images.detach()

# Sensi-CAM Heatmap Generation for Important Region Identification
def sensi_cam(feature_map, class_score):
    sensi_cam = torch.autograd.grad(outputs=class_score, inputs=feature_map,
                                    grad_outputs=torch.ones(class_score.size()).cuda(),
                                    create_graph=True, retain_graph=True)[0]
    weighted_avg = torch.mean(sensi_cam, dim=[2, 3], keepdim=True) * feature_map
    heatmap = torch.relu(torch.sum(weighted_avg, dim=1)).unsqueeze(1)
    return heatmap

# Extract Bounding Boxes and Crop the Image Using Sensi-CAM
def extract_bounding_boxes_and_crop(image, heatmap, threshold=0.5):
    heatmap_np = heatmap.detach().cpu().numpy()
    bbox = np.where(heatmap_np > threshold * heatmap_np.max())
    
    if bbox[0].size == 0 or bbox[1].size == 0:
        return image
    
    y_min, y_max = bbox[0].min(), bbox[0].max()
    x_min, x_max = bbox[1].min(), bbox[1].max()
    
    cropped_image = image[:, :, y_min:y_max, x_min:x_max]
    cropped_image = F.interpolate(cropped_image, size=(224, 224))
    
    return cropped_image

# Denoising (using Gaussian blur)
def denoise_image(image):
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    denoised_image_np = cv2.GaussianBlur(image_np, (5, 5), 0)
    denoised_image = torch.tensor(denoised_image_np.transpose(2, 0, 1)).float()
    return denoised_image.cuda()

# Extract Sobel and Canny Edges and Superimpose on the Original Image
def extract_edges(image):
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)

    canny_edges = cv2.Canny(np.uint8(gray_image), 100, 200)

    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX)
    superimposed_image = np.dstack((image_np, sobel_edges, canny_edges))

    superimposed_image_tensor = torch.tensor(superimposed_image.transpose(2, 0, 1)).float() / 255.0
    return superimposed_image_tensor.cuda()

# Combine RGB and Edge Information for Clean, Adversarial, and Cropped (Augmented) Images
def combine_rgb_edges(image, denoise=True):
    if denoise:
        image = denoise_image(image)
    return extract_edges(image)

# OSPA-based Loss Function
def ospa_loss(benign_pred, adversarial_pred, teacher_pred, sigma=0.5, c=1.0):
    """
    Calculates the OSPA discrepancy loss between the teacher model and both
    benign and adversarial predictions (heatmaps), incorporating x-center,
    y-center, and number of pixels, which are normalized.
    """
    
    def get_center_of_mass_and_pixels(heatmap):
        """
        Calculates the x and y center of mass and the number of pixels in a heatmap.
        The values are normalized between 0 and 1.
        """
        # Normalize the heatmap
        heatmap = heatmap / heatmap.sum(dim=[2, 3], keepdim=True)

        # Generate x and y coordinates
        x = torch.linspace(0, heatmap.shape[2] - 1, heatmap.shape[2]).cuda()
        y = torch.linspace(0, heatmap.shape[3] - 1, heatmap.shape[3]).cuda()

        # Calculate the center of mass
        x_center = (x * heatmap.sum(dim=3)).sum(dim=2)
        y_center = (y * heatmap.sum(dim=2)).sum(dim=2)

        # Calculate the number of pixels
        num_pixels = heatmap.sum(dim=[2, 3])

        # Normalize x, y centers and number of pixels between 0 and 1
        x_center_norm = x_center / heatmap.shape[2]
        y_center_norm = y_center / heatmap.shape[3]
        num_pixels_norm = num_pixels / (heatmap.shape[2] * heatmap.shape[3])

        return x_center_norm, y_center_norm, num_pixels_norm
    
    def ospa_distance(x1, x2, p=2):
        """Calculates OSPA distance between two feature sets."""
        dist = torch.norm(x1 - x2, p=p, dim=1)
        dist_clamped = torch.min(dist, torch.tensor(c).cuda())
        return dist_clamped

    # Extract centers and pixel counts for benign, adversarial, and teacher heatmaps
    x_benign, y_benign, pixels_benign = get_center_of_mass_and_pixels(benign_pred)
    x_adv, y_adv, pixels_adv = get_center_of_mass_and_pixels(adversarial_pred)
    x_teacher, y_teacher, pixels_teacher = get_center_of_mass_and_pixels(teacher_pred)

    # Calculate OSPA distances for x-center, y-center, and pixel counts
    dist_x_benign = ospa_distance(x_benign, x_teacher)
    dist_y_benign = ospa_distance(y_benign, y_teacher)
    dist_pixels_benign = ospa_distance(pixels_benign, pixels_teacher)

    dist_x_adv = ospa_distance(x_adv, x_teacher)
    dist_y_adv = ospa_distance(y_adv, y_teacher)
    dist_pixels_adv = ospa_distance(pixels_adv, pixels_teacher)

    # Combine the distances for benign and adversarial heatmaps
    ospa_benign = (dist_x_benign + dist_y_benign + dist_pixels_benign) / 3
    ospa_adv = (dist_x_adv + dist_y_adv + dist_pixels_adv) / 3

    # Calculate the discrepancy loss based on the OSPA distances
    discrepancy_loss = torch.exp(-((ospa_benign ** 2 + ospa_adv ** 2) / (2 * sigma ** 2)))

    return discrepancy_loss.mean()

# Combined Loss: Cross-Entropy + OSPA Loss
def combined_loss(output_benign, output_adv, target, benign_heatmap, adv_heatmap, teacher_heatmap, lambda_ospa=1.0):
    """
    Combines the cross-entropy loss with the OSPA discrepancy loss for model training.
    """

    # Cross-Entropy loss for benign images
    ce_loss_benign = F.cross_entropy(output_benign, target)

    # Cross-Entropy loss for adversarial images
    ce_loss_adv = F.cross_entropy(output_adv, target)

    # OSPA discrepancy loss between the student model's heatmaps (benign and adversarial) and the teacher model's heatmap
    ospa_discrepancy = ospa_loss(benign_heatmap, adv_heatmap, teacher_heatmap)

    # Combine the Cross-Entropy loss and OSPA discrepancy loss
    total_loss = ce_loss_benign + ce_loss_adv + lambda_ospa * ospa_discrepancy

    return total_loss

# Training Loop
def train(model, teacher_model, train_loader, optimizer, epochs=10, epsilon=0.03, num_steps=40, step_size=0.01, lambda_ospa=1.0):
    attack = LinfPGDAttack(model, epsilon=epsilon, num_steps=num_steps, step_size=step_size)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            # Generate adversarial examples using PGD attack
            adv_images = attack.perturb(images, labels)

            # Forward pass through the teacher model to get the teacher heatmap
            teacher_features = teacher_model.features(images)
            teacher_heatmap = sensi_cam(teacher_features, teacher_model(images).max(1)[0])

            # Forward pass for benign images through the student model
            benign_outputs = model(images)
            benign_features = model.features(images)
            benign_heatmap = sensi_cam(benign_features, benign_outputs.max(1)[0])

            # Forward pass for adversarial images through the student model
            adv_outputs = model(adv_images)
        


            adv_features = model.features(adv_images)
            adv_heatmap = sensi_cam(adv_features, adv_outputs.max(1)[0])

            # Compute the combined loss (cross-entropy + OSPA discrepancy)
            loss = combined_loss(benign_outputs, adv_outputs, labels, benign_heatmap, adv_heatmap, teacher_heatmap, lambda_ospa)
            total_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader)}')


