import torch
import torch.nn as nn
from models.utils import UnetConv, UnetUp_Concat
import torch.nn.functional as F
from models.network_helper import init_weights

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixerBlock(nn.Module):
    def __init__(self, dim=1024, depth=7, k=7):
        super(ConvMixerBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(dim, dim, kernel_size=(k, k), groups=dim, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )

    def forward(self, x):
        x = self.block(x)
        return x


class CMUnet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=4, in_channels=3,
                 mode='segmentation', model_kwargs=None):
        super(CMUnet, self).__init__()
        self.is_deconv = model_kwargs.is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = model_kwargs.is_batchnorm
        
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

        #ConvMixerBlock, block number depth=7, k=7
        self.ConvMixer = ConvMixerBlock(dim=filters[4], depth=7, k=7)

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
        self.segmentation = nn.Conv2d(self.in_channels, 1, 1)

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

        # ConvMixer as a part of encoder
        cm = self.ConvMixer(center)



        if self.mode == 'classification':
            pooled = self.global_pool(cm)
            pooled = torch.flatten(pooled, 1)
            classification_output = self.classifier(pooled)

            up4 = self.up_concat4(conv4, cm)
            up3 = self.up_concat3(conv3, up4)
            up2 = self.up_concat2(conv2, up3)
            up1 = self.up_concat1(conv1, up2)
            reconstruct = self.final(up1)

            return classification_output, reconstruct

        elif self.mode == 'segmentation':
            up4 = self.up_concat4(conv4, cm)
            up3 = self.up_concat3(conv3, up4)
            up2 = self.up_concat2(conv2, up3)
            up1 = self.up_concat1(conv1, up2)

            final = self.final(up1)
            final = self.segmentation(final)

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
        for param in self.ConvMixer.parameters():
            param.requires_grad = False

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
