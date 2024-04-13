import torch
from utils import *
from torch.utils.data import DataLoader
from tumour_cls_dataset import tumourClassificationDataset
from tumour_seg_dataset import tumourSegmentationDataset
from torch import optim
from models import *
from tqdm import tqdm
# import matplotlib.pyplot as plt


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
        for image, labels in train_loader:
            image, labels = image.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, reconstructions = model(image)
            loss = loss_fn(outputs, labels)
            loss += 0.5 * torch.mean((reconstructions - image)**2)  # Reconstruction loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * image.size(0)
            total_train_samples += image.size(0)

    average_train_loss = total_loss / total_train_samples

    # validate after one epoch of training
    results = validate(model, val_loader, loss_fn, device, task)
    if task == 'classification':
        average_val_loss, val_accuracy = results
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        model = myModel(model_opts.feature_scale, model_opts.n_classes, model_opts.is_deconv, model_opts.in_channels,
                        is_batchnorm=model_opts.is_batchnorm, mode=train_opts.task).to(device)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

    elif train_opts.task == 'segmentation':
        train_dataset = tumourSegmentationDataset(data_opts, 'train', transform_opts)
        val_dataset = tumourSegmentationDataset(data_opts, 'validation', transform_opts)
        train_loader = DataLoader(train_dataset, batch_size=data_opts.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=data_opts.test_batch_size, shuffle=False)

        myModel = get_model(model_opts.model_name)
        model = myModel(model_opts.feature_scale, model_opts.n_classes, model_opts.is_deconv, model_opts.in_channels,
                        is_batchnorm=model_opts.is_batchnorm, mode=train_opts.task).to(device)

        if train_opts.transfer_learning:
            try:
                save_path = os.path.join('saved_models', model_opts.pretrained_model)
                model.load_state_dict(torch.load(save_path)['model_state_dict'], strict=False) # initialize overlapping part
            except Exception as error:
                print('Caught this error when initialized pretrained model: ' + repr(error))

            # freeze the encoder part of the pretrained classification model
            model.freeze_encoder()

        loss_fn = torch.nn.BCEWithLogitsLoss()  # Can change Loss Function accordingly for segmentation task!!
        # initialize optimizer excluding frozen parameters
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, train_opts.epochs + 1):
        train_loss, val_loss = train(model, train_loader, val_loader, train_opts.task, optimizer, loss_fn, device, epoch, model_opts.save_model_name)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        print(f"Epoch: {epoch}; train loss: {train_loss:.4f}; validation loss: {val_loss:.4f}")

    # train_val_loss_plot(train_loss_list, val_loss_list)  # visualize training and validation loss
    print("Training of ", train_opts.task, " is finished!!!")


def train_val_loss_plot(train_loss_values, val_loss_values):
    # plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, label="Train Loss")
    # plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, label="Validation Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Train and Validation Loss vs. Number of Epochs")
    # plt.legend()
    # plt.show()
    pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='UNet Seg Transfer Learning Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    # Assuming you're running the classification task first
    main(args)
