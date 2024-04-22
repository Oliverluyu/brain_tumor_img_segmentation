import torch
from utils import *
from torch.utils.data import DataLoader
from tumour_cls_dataset import tumourClassificationDataset
from tumour_seg_dataset import tumourSegmentationDataset
from models import *
import matplotlib.pyplot as plt
import os


# Function to calculate Intersection over Union (IoU)
def calculate_iou(pred_mask, true_mask, threshold=0.5):
    pred_mask = (pred_mask > threshold).float()
    true_mask = (true_mask > threshold).float()

    intersection = torch.sum(pred_mask * true_mask)
    union = torch.sum((pred_mask + true_mask) > 0)

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def calculate_evaluation_metrics(pred_mask, true_mask, threshold=0.5):
    pred_mask = (pred_mask > threshold).float()
    true_mask = (true_mask > threshold).float()

    TP = torch.sum(pred_mask * true_mask).item()
    TN = torch.sum((1 - pred_mask) * (1 - true_mask)).item()
    FP = torch.sum(pred_mask * (1 - true_mask)).item()
    FN = torch.sum((1 - pred_mask) * true_mask).item()

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0  # also known as sensitivity
    specificity = TN / (TN + FP) if TN + FP > 0 else 0

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score, specificity


def visualize_images(image, mask, prediction, visualized_count, save_model_name, save_dir='saved_models'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ensure the image has 3 channels and convert it to display format
    image = image.squeeze().cpu().numpy()  # Remove batch dimension if present
    if image.ndim == 3 and image.shape[0] == 3:  # Check if it's channel-first and 3 channels
        image = image.transpose(1, 2, 0)  # Convert to channel-last format for matplotlib

    # Ensure the mask is grayscale and in correct format
    mask = mask.squeeze().cpu().numpy()
    prediction = prediction.squeeze().cpu().numpy()

    # Normalize the image if it's not in [0, 1] for display
    if image.min() < 0 or image.max() > 1:
        image = (image - image.min()) / (image.max() - image.min())

    # Visualize original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Visualize mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # Visualize prediction
    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    # Construct the filename incorporating the save_model_name
    filename = os.path.join(save_dir, f"{save_model_name}_{visualized_count + 1}.png")
    plt.savefig(filename)
    plt.close(fig)  # Close the plot to free up memory
    print(f"Saved visualization to {filename}")


def main(config_file):
    # Load configuration
    config = json_to_py_obj(config_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the test dataset
    data_opts = config.data
    test_dataset = tumourSegmentationDataset(data_opts, 'test', config.transform)
    test_loader = DataLoader(test_dataset, batch_size=data_opts.test_batch_size, shuffle=False)

    # Load the model
    model_opts = config.model
    model = get_model(model_opts.model_name)(model_opts.feature_scale, model_opts.n_classes, model_opts.in_channels,
                                             mode=config.training.task, model_kwargs=model_opts).to(device)
    model_path = os.path.join('saved_models', model_opts.save_model_name)

    # Adjust the map_location based on the availability of CUDA
    model_state = torch.load(model_path, map_location=device)  # Ensure the model loads onto the right device
    model.load_state_dict(model_state['model_state_dict'])
    model.eval()

    # Variables for IoU and F1 score calculations
    iou_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    specificity_scores = []
    visualized_count = 0

    # Evaluate the model
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            predictions = torch.sigmoid(model(images))

            for j, (img, msk, pred) in enumerate(zip(images, masks, predictions)):
                iou = calculate_iou(pred, msk)
                precision, recall, f1, specificity = calculate_evaluation_metrics(pred, msk)
                
                iou_scores.append(iou)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                specificity_scores.append(specificity)
            
                # Visualize and save the first few images, masks, and predictions
                if visualized_count < 3:  # Change this as needed
                    visualize_images(img, msk, pred, visualized_count, config.model.save_model_name, save_dir='saved_models')
                    visualized_count += 1

                # Break the loop if we have visualized the required number of images
                if visualized_count >= 3:
                    break

    # Calculate and print average IoU, F1 score
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_specificity = sum(specificity_scores) / len(specificity_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)
    average_iou = sum(iou_scores) / len(iou_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)

    print(f"Average IoU for the test set: {average_iou:.4f}")
    print(f"Average F1 Score for the test set: {average_f1:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Average Recall/Sensitivity: {average_recall:.4f}")
    print(f"Average Specificity: {average_specificity:.4f}")
    # print(f"Average F1 Score: {average_f1:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test model on segmentation dataset')
    parser.add_argument('-c', '--config', required=True, help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config)
