import torch
from utils import *
from torch import nn
from torch.utils.data import DataLoader
from tumour_cls_dataset import tumourClassificationDataset
from tumour_seg_dataset import tumourSegmentationDataset
from torch import optim
from models import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import os


# def train_val_loss_plot(train_loss_values, val_loss_values, val_accuracy_values):
#     epochs = range(1, len(train_loss_values) + 1)

#     plt.figure(figsize=(10, 5))

#     # Plot training and validation loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_loss_values, 'b', label='Training Loss')
#     plt.plot(epochs, val_loss_values, 'r', label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()

#     # Plot validation accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, val_accuracy_values, 'g', label='Validation Accuracy')
#     plt.title('Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()
def apply_transfer_learning(model, config):
    if config['training']['transfer_learning']:
        try:
            save_path = os.path.join('saved_models', config['model']['pretrained_model'])
            pretrained_dict = torch.load(save_path)
            new_state_dict = model.state_dict()
            transfer_layers = config['model']['transfer_layers']

            for key, value in pretrained_dict.items():
                if any(key.startswith(layer) for layer in transfer_layers):
                    if key in new_state_dict:
                        new_state_dict[key] = value
                        print(f"Transferred layer {key}")

            model.load_state_dict(new_state_dict, strict=False)
            print("Loaded pretrained weights for specified layers.")

        except Exception as error:
            print('Caught this error when initializing pretrained model: ' + repr(error))


def schedule_unfreezing(epoch, transfer_layers):
    # Start unfreezing only from the second epoch
    if epoch < 2:
        return []  # No layers to unfreeze during the first epoch
    # Ensure not to exceed the number of layers in transfer_layers
    num_layers_to_unfreeze = min(epoch - 1, len(transfer_layers))  # Adjust to start unfreezing from second epoch
    # Select the last `num_layers_to_unfreeze` layers from the list to unfreeze
    layers_to_unfreeze = transfer_layers[-num_layers_to_unfreeze:]
    return layers_to_unfreeze


def unfreeze_model_layers(model, layers_to_unfreeze):
    # Initially, freeze all layers
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
    # Unfreeze the specified layers
    for name, parameter in model.named_parameters():
        if any(name.startswith(layer) for layer in layers_to_unfreeze):
            parameter.requires_grad = True


class CombinedLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def dice_loss(self, logits, true):
        eps = 1e-7
        sigmoid = torch.sigmoid(logits)
        intersection = (sigmoid * true).sum((1, 2, 3))
        union = sigmoid.sum((1, 2, 3)) + true.sum((1, 2, 3))
        dice = (2. * intersection + eps) / (union + eps)
        return 1 - dice.mean()  # dice coefficient is between [0,1]; dice loss is 1 - Dice coefficient

    def forward(self, logits, true):
        loss_bce = self.bce_loss(logits, true)
        loss_dice = self.dice_loss(logits, true)
        return self.weight_bce * loss_bce + self.weight_dice * loss_dice


def validate(model, dataloader, loss_fn, device, task):
    model.eval()
    val_loss_sum = 0
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        if task == 'segmentation':
            for image, target in dataloader:
                image, target = image.to(device), target.to(device)
                output = model(image)
                loss = loss_fn(output, target)
                val_loss_sum += loss.item() * image.size(0)
                total_samples += image.size(0)

        elif task == 'classification':
            for image, labels in dataloader:
                image, labels = image.to(device), labels.to(device)
                output, _ = model(image)
                loss = loss_fn(output, labels)
                val_loss_sum += loss.item() * image.size(0)
                total_samples += image.size(0)

                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                correct_predictions += (predicted == labels).sum().item()

    average_val_loss = val_loss_sum / total_samples
    if task == 'classification':
        val_accuracy = 100 * correct_predictions / total_samples
        return average_val_loss, val_accuracy
    else:
        return average_val_loss


def train(model, train_loader, val_loader, task, optimizer, loss_fn, device, epoch, save_model_name):
    model.train()
    total_loss = 0
    total_train_samples = 0

    # save model path
    save_path = os.path.join('saved_models', save_model_name)
    best_eval_loss = torch.load(save_path)['eval_loss'] if os.path.exists(save_path) and epoch > 1 else float('inf')

    if task == 'segmentation':
        for image, target in tqdm(train_loader):
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * image.size(0)
            total_train_samples += image.size(0)

    elif task == 'classification':
        for image, labels in tqdm(train_loader):
            image, labels = image.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, reconstructions = model(image)
            loss = loss_fn(outputs, labels)
            loss += 0.5 * torch.mean((reconstructions - image) ** 2)  # Reconstruction loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * image.size(0)
            total_train_samples += image.size(0)

    average_train_loss = total_loss / total_train_samples

    # validate after one epoch of training
    results = validate(model, val_loader, loss_fn, device, task)
    if task == 'classification':
        average_val_loss, val_accuracy = results
        # print(f"Validation Accuracy: {val_accuracy:.2f}%")
    else:
        average_val_loss = results

    # checkpointing
    if average_val_loss < best_eval_loss:
        best_eval_loss = average_val_loss
        save_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "eval_loss": best_eval_loss,
            "epoch": epoch,
        }
        torch.save(save_dict, save_path)

    return average_train_loss, average_val_loss


def main(arguments):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')

    json_opts = json_to_py_obj(arguments.config)
    train_opts = json_opts.training
    model_opts = json_opts.model
    data_opts = json_opts.data
    transform_opts = json_opts.transform

    # TumourDataset = data_opts.dataset
    if train_opts.task == 'classification':
        train_dataset = tumourClassificationDataset(data_opts, 'train', transform_opts)
        val_dataset = tumourClassificationDataset(data_opts, 'validation', transform_opts)
        train_loader = DataLoader(train_dataset, batch_size=data_opts.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=data_opts.test_batch_size, shuffle=False)

        myModel = get_model(model_opts.model_name)
        model = myModel(model_opts.feature_scale, model_opts.n_classes, model_opts.in_channels,
                        mode=train_opts.task, model_kwargs=model_opts).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

    elif train_opts.task == 'segmentation':
        train_dataset = tumourSegmentationDataset(data_opts, 'train', transform_opts)
        val_dataset = tumourSegmentationDataset(data_opts, 'validation', transform_opts)
        train_loader = DataLoader(train_dataset, batch_size=data_opts.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=data_opts.test_batch_size, shuffle=False)

        myModel = get_model(model_opts.model_name)
        model = myModel(model_opts.feature_scale, model_opts.n_classes, model_opts.in_channels,
                        mode=train_opts.task, model_kwargs=model_opts).to(device)

        if train_opts.transfer_learning:
            print("applying transfer learning")
            try:
                save_path = os.path.join('saved_models', model_opts.pretrained_model)
                # model.load_state_dict(torch.load(save_path)['model_state_dict'], strict=False) # initialize overlapping part
                pretrained_dict = torch.load(save_path)
                '''
                pretrained_dict
                :model_state_dict
                :optimizer_state_dict
                :eval_loss
                :epoch
                '''
                # Access the model's state dictionary specifically
                pretrained_model_state = pretrained_dict['model_state_dict']
                new_state_dict = model.state_dict()
                transfer_layers = model_opts.transfer_layers
                frozen_layers = model_opts.frozen_layers
                # print (f" printing :{pretrained_dict.items()}")

                for key, value in pretrained_model_state.items():
                    # print (f" key :{key}")
                    if any(key.startswith(layer) for layer in transfer_layers):
                        if key in new_state_dict:
                            new_state_dict[key] = value
                            print(f"Transferred layer {key}")

                model.load_state_dict(new_state_dict, strict=False)
                print("Loaded pretrained weights for specified layers.")

                # If freezing is enabled, freeze the specified layers
                if train_opts.freeze:
                    # Freeze layers specified in the config file's frozen_layers
                    for name, parameter in model.named_parameters():
                        if any(name.startswith(layer) for layer in frozen_layers):
                            parameter.requires_grad = False
                    print(f"Frozen layers: {frozen_layers}")

            except Exception as error:
                print('Caught this error when initialized pretrained model: ' + repr(error))

            # # freeze the encoder part of the pretrained classification model
            # model.freeze_encoder()

        # loss_fn = nn.BCEWithLogitsLoss()  # Can change Loss Function accordingly for segmentation task!!
        loss_fn = CombinedLoss(weight_bce=0.5, weight_dice=0.5)
        # initialize optimizer excluding frozen parameters
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                    gamma=0.5)  # Decay LR by a factor of 0.5 every 10 epochs

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []  # Initialize list to store validation accuracies

    for epoch in range(1, train_opts.epochs + 1):

        if train_opts.task == 'segmentation' and train_opts.transfer_learning and train_opts.freeze and train_opts.gradually_unfreeze:
            layers_to_unfreeze = schedule_unfreezing(epoch, model_opts.frozen_layers)
            unfreeze_model_layers(model, layers_to_unfreeze)
            print(f"Epoch {epoch}: Unfreezing layers {layers_to_unfreeze}")

            # Check if there are any trainable parameters before reinitializing the optimizer
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if trainable_params:
                optimizer = optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-5)
                print("Optimizer reinitialized with newly unfrozen parameters.")
            else:
                print("No layers to unfreeze, optimizer not reinitialized.")

        train_loss, val_loss = train(model, train_loader, val_loader, train_opts.task, optimizer, loss_fn, device,
                                     epoch, model_opts.save_model_name)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        if train_opts.task == 'segmentation':
            scheduler.step()

        if train_opts.task == 'classification':
            _, val_accuracy = validate(model, val_loader, loss_fn, device, train_opts.task)
            val_accuracy_list.append(val_accuracy)
        print(f"Epoch: {epoch}; train loss: {train_loss:.4f}; validation loss: {val_loss:.4f}")

        if train_opts.task == 'classification':
            print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Plot losses and accuracy at the end of training
    if train_opts.task == 'classification':
        train_val_loss_plot(train_loss_list, val_loss_list, val_accuracy_list)
    else:
        train_val_loss_plot(train_loss_list, val_loss_list, [])  # Only plot loss for segmentation task

    print("Training of ", train_opts.task, " is finished!!!")


def train_val_loss_plot(train_loss_values, val_loss_values, val_accuracy_values=None,
                        save_model_name='model_performance', save_dir='saved_models'):
    epochs = range(1, len(train_loss_values) + 1)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{save_model_name}_{current_time}.png"
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_values, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss_values, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    # Plot validation accuracy if data is provided
    if val_accuracy_values:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_accuracy_values, 'g-', label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)  # Save the figure
    plt.show()
    print(f"Plot saved to {save_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='UNet Seg Transfer Learning Function')

    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    # Assuming you're running the classification task first
    main(args)
