import torch
from torch import nn
from einops import rearrange, reduce
from torch.nn import functional as F
from functools import partial
from .utils import GridAttentionBlock
from .network_helper import init_weights


# Helper functions and small modules (keep unchanged)
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
        )

    def forward(self, x):
        return self.up(x)


class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dim, default(dim_out, dim * 4), 3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.down(x)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class PreNorm(nn.Module):
    def __init__(self, num_groups, num_channels, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.fn(self.norm(x))


# Model components
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)  # (16, 4, 256, 32)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)  # (16, 128, 16, 16)
        return self.to_out(out)


class AttentionGateBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, sub_sample_factor):
        super(AttentionGateBlock, self).__init__()
        self.gate_block = GridAttentionBlock(in_channels=in_size, gating_channels=gate_size,
                                             inter_channels=inter_size, sub_sample_factor=sub_sample_factor)
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


class ConvMixerBlock(nn.Module):
    def __init__(self, dim=1024, depth=3, k=7):
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
        self.res_conv = nn.Identity()

    def forward(self, x):
        h = self.block(x)
        return h + self.res_conv(x)


class ProposedAttentionUnet(nn.Module):
    def __init__(self, dim, n_classes=4, in_channels=3, dim_mults=(1, 2, 4, 8, 16),  # channel: 16, 32, 64, 128, 256
                 attention_dsample=(2, 2), mode='segmentation', model_kwargs=None):
        super().__init__()
        self.channels = in_channels
        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding=3)
        self.groups = model_kwargs.resnet_block_groups
        self.mode = mode

        dims = [*map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=self.groups)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Downsampling blocks
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            blocks = nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Residual(PreNorm(self.groups, dim_in, LinearAttention(dim_in)))
            ])

            if is_last:
                blocks.append(nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1))
            else:
                blocks.append(Downsample(dim_in, dim_out))

            self.downs.append(blocks)

        # classifier head for pretraining on classification task
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(dims[-1], n_classes)

        # mid blocks
        mid_dim = dims[-1]
        # self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_block1 = ConvMixerBlock(mid_dim, depth=2, k=5)
        self.mid_attn = Residual(PreNorm(self.groups, mid_dim, Attention(mid_dim)))
        # self.mid_block2 = block_klass(mid_dim, mid_dim)
        self.mid_block2 = ConvMixerBlock(mid_dim, depth=2, k=7)

        # attention gate blocks
        self.attention_gate_block = nn.ModuleList([
            AttentionGateBlock(in_size=dims[2], gate_size=dims[4], inter_size=dims[2],
                               sub_sample_factor=attention_dsample),
            AttentionGateBlock(in_size=dims[1], gate_size=dims[3], inter_size=dims[1],
                               sub_sample_factor=attention_dsample),
            AttentionGateBlock(in_size=dims[0], gate_size=dims[2], inter_size=dims[0],
                               sub_sample_factor=attention_dsample)
        ])

        # Upsampling blocks
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            blocks = nn.ModuleList([
                block_klass(dim_in + dim_out, dim_out),
                block_klass(dim_in + dim_out, dim_out),
                Residual(PreNorm(self.groups, dim_out, LinearAttention(dim_out)))
            ])

            if is_last:
                blocks.append(nn.Conv2d(dim_out, dim_in, kernel_size=3, padding=1))
            else:
                blocks.append(Upsample(dim_out, dim_in))

            self.ups.append(blocks)

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, in_channels, 1)
        self.segmentation = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)
            x = block2(x)
            h.append(x)
            x = attn(x)
            x = downsample(x)

        if self.mode == 'classification':
            pooled = self.global_pool(x)
            pooled = torch.flatten(pooled, 1)
            classification_output = self.classifier(pooled)

            x = self.mid_block1(x)
            x = self.mid_attn(x)
            x = self.mid_block2(x)

            for block1, block2, attn, upsample in self.ups:
                x = torch.cat([x, h.pop()], dim=1)
                x = block1(x)
                x = torch.cat([x, h.pop()], dim=1)
                x = block2(x)
                x = attn(x)
                x = upsample(x)

            x = torch.cat((x, r), dim=1)

            x = self.final_res_block(x)
            reconstruct = self.final_conv(x)
            return classification_output, reconstruct

        elif self.mode == 'segmentation':

            x = self.mid_block1(x)
            x = self.mid_attn(x)
            x = self.mid_block2(x)

            gate = []

            for idx, (block1, block2, attn, upsample) in enumerate(self.ups):
                if idx == 0:
                    x = torch.cat([x, h.pop()], dim=1)
                else:
                    # attention gate for first block in each level
                    attention_gate = self.attention_gate_block[idx-1]
                    g_conv, _ = attention_gate(h.pop(), gate.pop())
                    x = torch.cat([x, g_conv], dim=1)

                x = block1(x)
                x = torch.cat([x, h.pop()], dim=1)
                x = block2(x)
                x = attn(x)
                gate.append(x)
                x = upsample(x)

            x = torch.cat((x, r), dim=1)

            x = self.final_res_block(x)
            x = self.final_conv(x)
            return self.segmentation(x)
