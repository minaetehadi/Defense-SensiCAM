import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import load_model  # Assuming load_model is your custom model loading function
from train import LinfPGDAttack, sensi_cam, extract_edges, combine_rgb_edges

def validate(model, teacher_model, validation_loader, epsilon=0.03, num_steps=40, step_size=0.01, lambda_ospa=1.0):

    model.eval()  # Set the model to evaluation mode
    total_examples = 0
    correct_benign = 0
    correct_adv = 0
    detected_adv = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    perturbation_size = 0

    attack = LinfPGDAttack(model, epsilon=epsilon, num_steps=num_steps, step_size=step_size)

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.cuda(), labels.cuda()
            total_examples += labels.size(0)

            outputs_benign = model(images)
            _, predicted_benign = torch.max(outputs_benign.data, 1)

            correct_benign += (predicted_benign == labels).sum().item()

            adv_images = attack.perturb(images, labels)

            # Forward pass for adversarial images
            outputs_adv = model(adv_images)
            _, predicted_adv = torch.max(outputs_adv.data, 1)

            correct_adv += (predicted_adv == labels).sum().item()

            detected_adv += (predicted_adv != labels).sum().item()


            for i in range(labels.size(0)):
                if predicted_benign[i] == labels[i] and predicted_adv[i] == labels[i]:
                    true_positive += 1
                elif predicted_benign[i] == labels[i] and predicted_adv[i] != labels[i]:
                    true_negative += 1
                elif predicted_benign[i] != labels[i] and predicted_adv[i] == labels[i]:
                    false_positive += 1
                else:
                    false_negative += 1

    benign_accuracy = 100 * correct_benign / total_examples
    adv_accuracy = 100 * correct_adv / total_examples
    dsr = 100 * correct_adv / total_examples  # Defense Success Rate
    dr = 100 * detected_adv / total_examples  # Detection Rate
    avg_perturbation = perturbation_size / total_examples

    classification_accuracy = 100 * (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    print(f'Benign Accuracy: {benign_accuracy:.2f}%')
    print(f'Adversarial Accuracy: {adv_accuracy:.2f}%')
    print(f'Defense Success Rate (DSR): {dsr:.2f}%')
    print(f'Detection Rate (DR): {dr:.2f}%')
    print(f'Average Perturbation Size: {avg_perturbation:.4f}')
    print(f'Classification Accuracy (TP, TN, FP, FN): {classification_accuracy:.2f}%')

    return benign_accuracy, adv_accuracy, dsr, dr, avg_perturbation

def visualize_sensi_cam(model, images, labels):

    model.eval()
    heatmaps = []
    
    with torch.no_grad():
        # Forward pass to get feature maps and outputs
        features = model.features(images)
        outputs = model(images)
        class_scores = outputs.max(1)[0]

        for i in range(len(images)):
            heatmap = sensi_cam(features[i].unsqueeze(0), class_scores[i].unsqueeze(0))
            heatmaps.append(heatmap)

    return heatmaps


def visualize_results(images, heatmaps, title="Sensi-CAM Heatmaps"):
    fig, axs = plt.subplots(2, len(images), figsize=(15, 5))

    for i in range(len(images)):
        axs[0, i].imshow(images[i].cpu().permute(1, 2, 0).numpy())
        axs[0, i].set_title(f'Image {i+1}')
        axs[0, i].axis('off')

        heatmap_np = heatmaps[i].cpu().numpy().squeeze()
        axs[1, i].imshow(heatmap_np, cmap='hot')
        axs[1, i].set_title(f'Heatmap {i+1}')
        axs[1, i].axis('off')

    plt.suptitle(title)
    plt.show()


def run_test(model, teacher_model, test_loader, epsilon=0.03, num_steps=40, step_size=0.01, lambda_ospa=1.0):

    print("Running validation on both clean and adversarial examples...")
    benign_acc, adv_acc, dsr, dr, avg_perturbation = validate(model, teacher_model, test_loader, epsilon, num_steps, step_size, lambda_ospa)

    # Visualize heatmaps for a batch of clean and adversarial examples
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()

        print("\nVisualizing Sensi-CAM heatmaps for clean examples...")
        clean_heatmaps = visualize_sensi_cam(model, images, labels)
        visualize_results(images.cpu(), clean_heatmaps, title="Clean Image Heatmaps")

        print("\nVisualizing Sensi-CAM heatmaps for adversarial examples...")
        attack = LinfPGDAttack(model, epsilon=epsilon, num_steps=num_steps, step_size=step_size)
        adv_images = attack.perturb(images, labels)
        adv_heatmaps = visualize_sensi_cam(model, adv_images, labels)
        visualize_results(adv_images.cpu(), adv_heatmaps, title="Adversarial Image Heatmaps")
        break

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = load_model(model_name='VGG19').cuda()  # Student model
    teacher_model = load_model(model_name='VGG19').cuda()  # Pre-trained teacher model

    run_test(model, teacher_model, test_loader)
