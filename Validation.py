import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import load_model
from train import LinfPGDAttack, sensi_cam, extract_edges, combine_rgb_edges

def validate(model, teacher_model, validation_loader, epsilon=0.03, num_steps=40, step_size=0.01):
    model.eval()  # Set the model to evaluation mode
    total_examples = 0
    correct_benign = 0
    correct_adv = 0
    detected_adv = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    perturbations = []

    attack = LinfPGDAttack(model, epsilon=epsilon, num_steps=num_steps, step_size=step_size)

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.cuda(), labels.cuda()
            total_examples += labels.size(0)

            outputs_benign = model(images)
            _, predicted_benign = torch.max(outputs_benign.data, 1)

            adv_images = attack.perturb(images, labels)
            outputs_adv = model(adv_images)
            _, predicted_adv = torch.max(outputs_adv.data, 1)


            correct_adv += (predicted_adv == labels).sum().item()

            detected_adv += (predicted_adv != labels).sum().item()



    accuracy = 100 * (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    dsr = 100 * correct_adv / total_examples

    dr = 100 * detected_adv / total_examples

    print(f'Defense Success Rate (DSR): {dsr:.2f}%')
    print(f'Detection Rate (DR): {dr:.2f}%')
    print(f'Average Perturbation Size: {avg_perturbation:.4f}')


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = load_model(model_name='VGG19').cuda()
    teacher_model = load_model(model_name='VGG19').cuda()  # Load your teacher model

    validate(model, teacher_model, val_loader)
