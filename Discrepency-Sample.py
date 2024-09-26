import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from scipy.optimize import linear_sum_assignment

def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_sensicam_heatmap(model, img_array, layer_name='block5_conv4', class_idx=5):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_channel = predictions[:, class_idx]  # Target class

    grads = tape.gradient(class_channel, conv_output)[0]
    conv_output = conv_output[0]
    grads = grads.numpy()
    conv_output = conv_output.numpy()

    weights = np.mean(grads, axis=(0, 1))
    cam = np.dot(conv_output, weights)

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))
    cam = cam / np.max(cam)
    return cam

def overlay_heatmap_on_image(heatmap, original_img):
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
    return overlayed_img

def display_heatmap_overlay(img_path, heatmap):
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))

    overlayed_img = overlay_heatmap_on_image(heatmap, original_img)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))
    plt.title('SensiCAM Heatmap Overlay')
    plt.axis('off')
    plt.show()

def segment_image(heatmap, threshold=0.5):
    max_value = np.max(heatmap)
    binary_mask = heatmap > (threshold * max_value)
    return binary_mask

def compute_center_and_size(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    centers, sizes = [], []
    for i in range(1, num_labels):
        center_x, center_y = centroids[i]
        area = stats[i, cv2.CC_STAT_AREA]
        centers.append((center_x, center_y))
        sizes.append(area)
    return centers, sizes

def normalize_centers_and_sizes(centers, sizes, img_width, img_height):
    normalized_centers = [(x / img_width, y / img_height) for x, y in centers]
    total_pixels = img_width * img_height
    normalized_sizes = [size / total_pixels for size in sizes]
    return normalized_centers, normalized_sizes

def compute_ospa(centers1, sizes1, centers2, sizes2, c=1.0):
    n1, n2 = len(centers1), len(centers2)

    if n1 == 0 and n2 == 0:
        return 0.0
    if n1 == 0 or n2 == 0:
        return c

    dist_matrix = np.sqrt(np.sum((np.array(centers1)[:, np.newaxis] - np.array(centers2)) ** 2, axis=2))
    size_diff_matrix = np.abs(np.array(sizes1)[:, np.newaxis] - np.array(sizes2))
    total_cost_matrix = dist_matrix + size_diff_matrix * c

    row_ind, col_ind = linear_sum_assignment(total_cost_matrix)
    assignment_cost = total_cost_matrix[row_ind, col_ind].sum()

    miss_cost1 = c * (n1 - len(row_ind))
    miss_cost2 = c * (n2 - len(col_ind))

    return (assignment_cost + miss_cost1 + miss_cost2) / max(n1, n2)

def display_segmented_image(img_path, segmented_img, centers, sizes):
    plt.figure()
    plt.imshow(segmented_img)
    plt.title('Segmented Image')
    plt.axis('off')

    for (x, y), size in zip(centers, sizes):
        plt.scatter(x, y, c='red', s=100, marker='x')
        plt.text(x, y, f'{size:.2f}\n({x:.2f}, {y:.2f})', color='red', fontsize=8, ha='right',
                 bbox=dict(facecolor='white', alpha=0.5))

    plt.show()
  
def print_centers_and_sizes(normalized_centers, normalized_sizes, img_idx):
    for i, ((x, y), size) in enumerate(zip(normalized_centers, normalized_sizes)):
        print(f"Image {img_idx + 1}, Segment {i + 1}:")
        print(f"  xCenter (normalized): {x:.2f}")
        print(f"  yCenter (normalized): {y:.2f}")
        print(f"  Number of Pixels (normalized): {size:.2f}")

def process_images(model, img_path1, img_path2, target_size=(224, 224)):
    img_array1 = load_and_preprocess_image(img_path1, target_size)
    img_array2 = load_and_preprocess_image(img_path2, target_size)

    heatmap1 = get_sensicam_heatmap(model, img_array1, class_idx=5)
    heatmap2 = get_sensicam_heatmap(model, img_array2, class_idx=5)

    display_heatmap_overlay(img_path1, heatmap1)

    binary_mask1 = segment_image(heatmap1)
    binary_mask2 = segment_image(heatmap2)

    original_img1 = cv2.imread(img_path1)
    original_img2 = cv2.imread(img_path2)
    original_img1 = cv2.resize(original_img1, target_size)
    original_img2 = cv2.resize(original_img2, target_size)

    segmented_img1 = np.zeros_like(original_img1)
    segmented_img2 = np.zeros_like(original_img2)

    for c in range(3):
        segmented_img1[:, :, c] = original_img1[:, :, c] * binary_mask1
        segmented_img2[:, :, c] = original_img2[:, :, c] * binary_mask2

    centers1, sizes1 = compute_center_and_size(binary_mask1)
    centers2, sizes2 = compute_center_and_size(binary_mask2)

    img_width, img_height = original_img1.shape[1], original_img1.shape[0]
    normalized_centers1, normalized_sizes1 = normalize_centers_and_sizes(centers1, sizes1, img_width, img_height)
    normalized_centers2, normalized_sizes2 = normalize_centers_and_sizes(centers2, sizes2, img_width, img_height)


    print_centers_and_sizes(normalized_centers1, normalized_sizes1, 0)
    print_centers_and_sizes(normalized_centers2, normalized_sizes2, 1)

    display_segmented_image(img_path1, cv2.cvtColor(segmented_img1, cv2.COLOR_BGR2RGB), normalized_centers1, normalized_sizes1)
    display_segmented_image(img_path2, cv2.cvtColor(segmented_img2, cv2.COLOR_BGR2RGB), normalized_centers2, normalized_sizes2)

    ospa_dist = compute_ospa(normalized_centers1, normalized_sizes1, normalized_centers2, normalized_sizes2)
    print(f"OSPA Distance between the two segmented images: {ospa_dist}")

model = VGG19(weights='imagenet')

# Paths to two images
img_path1 = '/content/dog1.jpg'
img_path2 = '/content/dog2.jpg'

process_images(model, img_path1, img_path2)
