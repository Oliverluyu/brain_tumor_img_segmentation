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

class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.up(x)
        return x

class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.conv(x)
        return x

class CMUNeXt(nn.Module):

    def __init__(self, feature_scale=4, n_classes=4, in_channels=3,
                 mode='segmentation', model_kwargs=None):
        super(CMUNeXt, self).__init__()
        self.is_deconv = model_kwargs.is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = model_kwargs.is_batchnorm
        
        self.feature_scale = feature_scale
        self.mode = mode

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        depths=[1, 1, 1, 3, 1]
        kernels=[3, 3, 7, 7, 7]
        # encoder for feature extraction
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=in_channels, ch_out=filters[0])
        self.encoder1 = CMUNeXtBlock(ch_in=filters[0], ch_out=filters[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=filters[0], ch_out=filters[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=filters[1], ch_out=filters[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=filters[2], ch_out=filters[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=filters[3], ch_out=filters[4], depth=depths[4], k=kernels[4])

        # classifier head for pretraining on classification task
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(filters[4], n_classes)

        # Decoder
        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Up_conv5 = fusion_conv(ch_in=filters[3] * 2, ch_out=filters[3])
        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Up_conv4 = fusion_conv(ch_in=filters[2] * 2, ch_out=filters[2])
        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Up_conv3 = fusion_conv(ch_in=filters[1] * 2, ch_out=filters[1])
        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Up_conv2 = fusion_conv(ch_in=filters[0] * 2, ch_out=filters[0])
        self.Conv_1x1 = nn.Conv2d(filters[0], 1, kernel_size=1, stride=1, padding=0)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        if self.mode == 'classification':
            pooled = self.global_pool(x5)
            pooled = torch.flatten(pooled, 1)
            classification_output = self.classifier(pooled)
           
            d5 = self.Up5(x5)
            d5 = torch.cat((x4, d5), dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)

            d1 = self.Conv_1x1(d2)
            
            return classification_output, d1

        elif self.mode == 'segmentation':
            d5 = self.Up5(x5)
            d5 = torch.cat((x4, d5), dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)

            d1 = self.Conv_1x1(d2)

            return d1

    def freeze_encoder(self):
        """freeze the encoder weights, for transfer learning to segmentation task."""
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for param in self.encoder3.parameters():
            param.requires_grad = False
        for param in self.encoder4.parameters():
            param.requires_grad = False
        for param in self.encoder5.parameters():
            param.requires_grad = False

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p