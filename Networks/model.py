import os
import sys
import torch
from torch import nn

base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config


class ResBlock(nn.Sequential):
    def __init__(self, in_channels=256, mid_channels=512, out_channels=256, kernel_size=3):
        conv1 = nn.Conv1d(
            in_channels, mid_channels * 2,
            kernel_size, 1, "same",
        )
        norm1 = nn.InstanceNorm1d(mid_channels * 2)
        GLU = nn.GLU(dim=1)
        conv2 = nn.Conv1d(
            mid_channels, out_channels,
            kernel_size, 1, "same",
        )
        norm2 = nn.InstanceNorm1d(out_channels)
        super().__init__(conv1, norm1, GLU, conv2, norm2)

    def forward(self, input):
        output = super().forward(input)
        output = output + input
        return output


class DownSample(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Conv2d(in_channels, out_channels*2,
                         kernel_size, stride, padding)
        norm = nn.InstanceNorm2d(out_channels * 2)
        GLU = nn.GLU(dim=1)
        super().__init__(conv, norm, GLU)


class UpSample(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upscale_factor=2):
        conv = nn.Conv2d(in_channels, out_channels*2*(upscale_factor**2),
                         kernel_size, stride, padding)
        PS = nn.PixelShuffle(upscale_factor)
        norm = nn.InstanceNorm2d(out_channels*2)
        GLU = nn.GLU(dim=1)
        super().__init__(conv, PS, norm, GLU)


class Reshape2dto1d(nn.Sequential):
    def __init__(self, in_channels, in_features, out_channels):
        self.in_channels, self.in_features, self.out_channels = in_channels, in_features, out_channels
        conv = nn.Conv1d(in_channels*in_features, out_channels, 1)
        norm = nn.InstanceNorm1d(out_channels)
        super().__init__(conv, norm)

    def forward(self, input: torch.Tensor):
        n, c, f, t = input.shape
        assert (c == self.in_channels and f == self.in_features), \
            f"Input dimension (n, c, t) should satisfy (c == in_channels and f == in_features)"
            
        output = input.reshape((n, c*f, t))
        output = super().forward(output)
        return output


class Reshape1dto2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, out_features):
        self.in_channels, self.out_channels, self.out_features = in_channels, out_channels, out_features
        conv = nn.Conv1d(in_channels, out_channels*out_features, 1)
        norm = nn.InstanceNorm1d(out_channels*out_features)
        super().__init__(conv, norm)

    def forward(self, input: torch.Tensor):
        n, c, t = input.shape
        assert (c == self.in_channels), \
            f"Input dimension (n, c, t) should satisfy (c == in_channels)"

        output = super().forward(input)
        output = output.reshape((n, self.out_channels, self.out_features, t))
        return output


class Generator(nn.Sequential):
    def __init__(self):
        input = [
            nn.Conv2d(2, 256, (5, 15), 1, (2, 7)),
            nn.GLU(dim=1),
        ]
        downsample = [
            DownSample(128, 256, 3, 2, 1),
            DownSample(256, 512, 3, 2, 1),
        ]
        reshape2dto1d = [
            Reshape2dto1d(512, config.N_MELS//4, 256),
        ]
        resblock = [
            ResBlock() for _ in range(6)
        ]
        reshape1dto2d = [
            Reshape1dto2d(256, 256, config.N_MELS//4),
        ]
        upsample = [
            UpSample(256, 256, 5, 1, 2, 2),
            UpSample(256, 128, 5, 1, 2, 2),
        ]
        output = [
            nn.Conv2d(128, 1, (5, 15), 1, (2, 7)),
        ]
        layers = \
            input + downsample + reshape2dto1d + resblock + reshape1dto2d + upsample + output
        super().__init__(*layers)


class Discriminator(nn.Sequential):
    def __init__(self):
        input = [
            nn.Conv2d(1, 256, 3, 1),
            nn.GLU(dim=1),
        ]
        downsample = [
            DownSample(128,  256,  3, 2, 1),
            DownSample(256,  512,  3, 2, 1),
            DownSample(512,  1024, 3, 2, 1),
            DownSample(1024, 1024, (1, 5), 1, (0, 2)),
        ]
        output = [
            nn.Conv2d(1024, 1, (1, 3), 1, (0, 1)),
            nn.Sigmoid()
        ]
        layers = \
            input + downsample + output
        super().__init__(*layers)


if __name__ == '__main__':
    import os
    import sys
    base_folder = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    if base_folder not in sys.path:
        sys.path.append(base_folder)
    if True:
        import config
        from Data import VCDataset, VCDataLoader

    Gxy, Gyx = Generator().to(config.device), Generator().to(config.device)
    Dx, Dy = Discriminator().to(config.device), Discriminator().to(config.device)
    
    # Trivial test for dataset class
    trainX = torch.load(os.path.abspath(os.path.join(
        base_folder, config.preprocessed_train_data, "./Ikura/Ikura.data"
    )))
    trainY = torch.load(os.path.abspath(os.path.join(
        base_folder, config.preprocessed_train_data, "./Trump/Trump.data"
    )))
    dataset = VCDataset(trainX, trainY).to(config.device)
    dataloader = VCDataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    while(True):
        for i, (X, maskX, Y, maskY) in enumerate(dataloader):
            print(Dy(Gxy(torch.cat((X, maskX), dim=1))).shape)
            print(Dy(Y).shape)
            print(Dx(Gyx(torch.cat((Y, maskY), dim=1))).shape)
            print(Dx(X).shape)