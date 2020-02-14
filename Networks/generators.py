import torch.nn as nn
import torch.nn.functional as F
import torch


def get_noise_generator(args):
    if args.ng_type == '1x1':
        noise_gen = NoiseResNet1x1Conv(args.channels, args.custom_init_ng)
    elif args.ng_type == '3x3':
        noise_gen = NoiseResNet3x3Conv(args.channels, args.custom_init_ng)
    else:
        raise Exception(f'generator type: {args.ng_type} is not available')
    noise_gen.epsilon = nn.Parameter(torch.tensor([args.epsilon_generator]), requires_grad=False)
    noise_gen.to(args.device)
    noise_gen.train()
    
    return noise_gen


class NoiseResNet1x1Conv(nn.Module):
    def __init__(self, channels, custom_init=True):
        super().__init__()
        self.conv_2d_1 = nn.Conv2d(in_channels=channels, out_channels=20, kernel_size=1, stride=1,
                                   padding=0)
        self.conv_2d_2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=1, stride=1,
                                   padding=0)
        self.conv_2d_3 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=1, stride=1,
                                   padding=0)
        self.conv_2d_4 = nn.Conv2d(in_channels=20, out_channels=channels, kernel_size=1, stride=1,
                                   padding=0)
        if custom_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.01)
                    m.bias.data.zero_()

    def forward(self, x):
        bs, ch, nx, ny = x.shape
        x = torch.empty((bs, ch, nx, ny), device=x.device).normal_()

        residual = x
        x = F.relu(self.conv_2d_1(x))
        x = F.relu(self.conv_2d_2(x))
        x = F.relu(self.conv_2d_3(x))
        x = self.conv_2d_4(x) + residual

        return x


class NoiseResNet3x3Conv(nn.Module):
    def __init__(self, channels, custom_init=True):
        super().__init__()
        self.conv_2d_1 = nn.Conv2d(in_channels=channels, out_channels=20, kernel_size=1, stride=1,
                                   padding=0)
        self.conv_2d_2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1,
                                   padding=0)
        self.conv_2d_3 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=1, stride=1,
                                   padding=0)
        self.conv_2d_4 = nn.Conv2d(in_channels=20, out_channels=channels, kernel_size=1, stride=1,
                                   padding=0)
 
        if custom_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.01)
                    m.bias.data.zero_()

    def forward(self, x):
        bs, ch, nx, ny = x.shape
        x = torch.empty((bs, ch, nx + 2, ny + 2), device=x.device).normal_()
        residual = x[:, :, 1:-1, 1:-1]

        x = F.relu(self.conv_2d_1(x))
        x = F.relu(self.conv_2d_2(x))
        x = F.relu(self.conv_2d_3(x))
        x = self.conv_2d_4(x) + residual

        return x    
