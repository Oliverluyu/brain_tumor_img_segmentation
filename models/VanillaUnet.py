import torch
import torch.nn as nn
from models.utils import UnetConv, UnetUp_Concat
import torch.nn.functional as F
from models.network_helper import init_weights


class unet_2D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=4, is_deconv=True, in_channels=3, attention_dsample=None,
                 is_batchnorm=True, mode='segmentation'):
        super(unet_2D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.mode = mode

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # encoder for feature extraction
        # downsampling
        self.conv1 = UnetConv(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UnetConv(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UnetConv(filters[3], filters[4], self.is_batchnorm)

        # classifier head for pretraining on classification task
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(filters[4], n_classes)


        # decoder for segmentation
        # upsampling
        self.up_concat4 = UnetUp_Concat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UnetUp_Concat(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp_Concat(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp_Concat(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], self.in_channels, 1)
        self.segmentation = nn.Conv2d(self.in_channels, 1,1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        if self.mode == 'classification':
            pooled = self.global_pool(center)
            pooled = torch.flatten(pooled, 1)
            classification_output = self.classifier(pooled)
            up4 = self.up_concat4(conv4, center)
            up3 = self.up_concat3(conv3, up4)
            up2 = self.up_concat2(conv2, up3)
            up1 = self.up_concat1(conv1, up2)
            final = self.final(up1)

            return classification_output, final

        elif self.mode == 'segmentation':
            up4 = self.up_concat4(conv4, center)
            up3 = self.up_concat3(conv3, up4)
            up2 = self.up_concat2(conv2, up3)
            up1 = self.up_concat1(conv1, up2)

            final = self.final(up1)
            # print(final.shape)
            # final = self.segmentation(final)
            # print(final.shape)

            return final

    def freeze_encoder(self):
        """freeze the encoder weights, for transfer learning to segmentation task."""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.conv3.parameters():
            param.requires_grad = False
        for param in self.conv4.parameters():
            param.requires_grad = False
        for param in self.center.parameters():
            param.requires_grad = False

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
