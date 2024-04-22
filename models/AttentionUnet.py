import torch.nn as nn
import torch
from .utils import UnetConv, UnetUp_Concat, UnetGridGatingSignal, GridAttentionBlock
import torch.nn.functional as F
from .network_helper import init_weights


class unet_CT_single_att(nn.Module):

    def __init__(self, feature_scale=4, n_classes=4, in_channels=3,
                 attention_dsample=(2,2), mode='segmentation', model_kwargs=None):
        super(unet_CT_single_att, self).__init__()
        self.is_deconv = model_kwargs.is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = model_kwargs.is_batchnorm
        self.feature_scale = feature_scale
        self.mode = mode

        filters = [64, 128, 256, 512, 1024]
        filters = [int(ft / self.feature_scale) for ft in filters]

        # downsampling
        self.conv1 = UnetConv(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3), padding_size=(1,1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = UnetConv(filters[0], filters[1], self.is_batchnorm, kernel_size=(3,3), padding_size=(1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = UnetConv(filters[1], filters[2], self.is_batchnorm, kernel_size=(3,3), padding_size=(1,1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = UnetConv(filters[2], filters[3], self.is_batchnorm, kernel_size=(3,3), padding_size=(1,1))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = UnetConv(filters[3], filters[4], self.is_batchnorm, kernel_size=(3,3), padding_size=(1,1))
        self.gating = UnetGridGatingSignal(filters[4], filters[4], kernel_size=(1, 1), is_batchnorm=self.is_batchnorm)

        # classifier head for pretraining on classification task
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(filters[4], n_classes)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   sub_sample_factor=attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   sub_sample_factor=attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp_Concat(filters[4], filters[3], self.is_batchnorm)
        self.up_concat3 = UnetUp_Concat(filters[3], filters[2], self.is_batchnorm)
        self.up_concat2 = UnetUp_Concat(filters[2], filters[1], self.is_batchnorm)
        self.up_concat1 = UnetUp_Concat(filters[1], filters[0], self.is_batchnorm)

        # deep supervision
        # self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        # self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        # self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        # self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)
        # final conv (without any concat)
        # self.final = nn.Conv2d(n_classes*4, n_classes, 1)

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
        # Feature Extraction (Encoder)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Getting gating signal
        center = self.center(maxpool4)
        gating = self.gating(center)

        if self.mode == 'classification':
            pooled = self.global_pool(gating)
            pooled = torch.flatten(pooled, 1)
            classification_output = self.classifier(pooled)
            return classification_output

        elif self.mode == 'segmentation':
            # Upscaling Part (Decoder)
            # Skip connection with attention gates
            g_conv4, att4 = self.attentionblock4(conv4, gating)
            up4 = self.up_concat4(g_conv4, center)
            g_conv3, att3 = self.attentionblock3(conv3, up4)
            up3 = self.up_concat3(g_conv3, up4)
            g_conv2, att2 = self.attentionblock2(conv2, up3)
            up2 = self.up_concat2(g_conv2, up3)
            up1 = self.up_concat1(conv1, up2)

            # Deep Supervision
            # dsv4 = self.dsv4(up4)
            # dsv3 = self.dsv3(up3)
            # dsv2 = self.dsv2(up2)
            # dsv1 = self.dsv1(up1)
            # final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))

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
        for param in self.gating.parameters():
            param.requires_grad = False

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block = GridAttentionBlock(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, sub_sample_factor= sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gated_feature, attention = self.gate_block(input, gating_signal)

        return self.combine_gates(gated_feature), attention