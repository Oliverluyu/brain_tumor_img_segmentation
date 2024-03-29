import torch
from utils import *
from torch.utils.data import DataLoader
from tumour_cls_dataset import tumourClassificationDataset
from tumour_seg_dataset import tumourSegmentationDataset
from torch import optim
from models import *
from tqdm import tqdm


def validate(model, dataloader, loss_fn, device):
    model.eval()
    val_loss_list = []
    val_acc_list = []

    with torch.no_grad():
        for image, target in dataloader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = loss_fn(output, target)
            val_loss_list.append(loss.item())

            # acculate accuracy
            # _, predicts = torch.max(output, dim=1)
            # acc = torch.sum(predicts == target).item() / len(target)
            # val_acc_list.append(acc)

    return np.mean(val_loss_list)


def train(model, train_loader, val_loader, optimizer, loss_fn, device, epoch, save_model_name):
    model.train()
    total_loss = 0
    step = 0
    # save model
    save_path = os.path.join('saved_models', save_model_name)
    best_eval_loss = torch.load(save_path)['eval_loss'] if epoch > 1 else 9999

    for image, target in tqdm(train_loader):
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        step += 1

        if step % 50 == 0:
            last_loss = total_loss / 50  # train loss per batch
            print(f"\nepoch: {epoch}, step: {step}, train loss: {last_loss}")
            total_loss = 0.

            curr_eval_loss = validate(model, val_loader, loss_fn, device)
            print(f"epoch: {epoch}, step: {step}, eval loss: {curr_eval_loss}")

            # early stop
            if curr_eval_loss < best_eval_loss:
                best_eval_loss = curr_eval_loss

                # put eval loss and acc in model state dict
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_loss": curr_eval_loss,
                    "epoch": epoch,
                }
                torch.save(save_dict, save_path)
            # switch to train mode
            model.train()


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

        MyModel = model_opts.model_name
        model = MyModel(model_opts.feature_scale, model_opts.n_classes, model_opts.is_deconv, model_opts.in_channels,
                        is_batchnorm=model_opts.is_batchnorm, mode=train_opts.task).to(device)
        try:
            save_path = os.path.join('saved_models', model_opts.pretrained_model)
            model.load_state_dict(torch.load(save_path)['model_state_dict'], strict=False)
        except Exception as error:
            print('Caught this error when initialized pretrained model: ' + repr(error))

        # freeze the encoder part of the pretrained classification model
        model.freeze_encoder()

        loss_fn = torch.nn.CrossEntropyLoss()  # Change Loss Function accordingly for segmentation task!!
        # initialize optimizer excluding frozen parameters
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    for epoch in range(1, train_opts.epochs + 1):
        print("\nEpoch :", epoch)
        train(model, train_loader, val_loader, optimizer, loss_fn, device, epoch, model_opts.save_model_name)

    print("Training of ", train_opts.task, " is finished!!!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='UNet Seg Transfer Learning Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    # Assuming you're running the classification task first
    main(args)
