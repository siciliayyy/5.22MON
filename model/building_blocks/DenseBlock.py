import torch
from torch import nn


# class DenseBlock(nn.Module):
#     """
#     Repeatable Dense block as specified by the paper
#     This is composed of a pointwise convolution followed by a depthwise separable convolution
#     After each convolution is a BatchNorm followed by a ReLU
#
#     Some notes on architecture based on the paper:
#       - The first block uses an input channel of 96, and the remaining input channels are 32
#       - The hidden channels is always 128
#       - The output channels is always 32
#       - The depth is always 3
#     """
#
#     def __init__(
#         self, in_channels: int, hidden_channels: int, out_channels: int, count: int
#     ):
#         """
#         Create the layers for the dense block
#
#         :param in_channels:      number of input features to the block
#         :param hidden_channels:  number of output features from the first convolutional layer   # 使其变厚，再变薄，压缩channel使得不会爆
#         :param out_channels:     number of output features from this entire block       #hidden channel确定中间
#         :param count:            number of times to repeat
#         """
#         super().__init__()
#
#         # First iteration takes different number of input channels and does not repeat
#         first_block = [
#             nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 1, 1)),
#             nn.BatchNorm3d(hidden_channels),        # 输入输出进行归一化，可有效防止梯度爆炸和梯度消失，能加快网络的收敛速度
#             nn.ReLU(),
#             nn.Conv3d(
#                 hidden_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
#             ),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(),
#         ]
#
#         # Remaining repeats are identical blocks 重复一致的block
#         repeating_block = [
#             nn.Conv3d(out_channels, hidden_channels, kernel_size=(1, 1, 1)),
#             nn.BatchNorm3d(hidden_channels),
#             nn.ReLU(),
#             nn.Conv3d(
#                 hidden_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
#             ),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(),
#         ]
#
#         self.convs = nn.Sequential(
#             *first_block,
#             *repeating_block * (count - 1),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the block
#
#         :param x:  image tensor
#         :return:   output of the forward pass
#         """
#         return self.convs(x)


class DenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.in_ch2 = in_ch + out_ch
        self.out_ch2 = out_ch * 3
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.in_ch2, self.out_ch2, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(self.out_ch2),
            nn.ReLU(inplace=True)
        )
        self.in_ch3 = self.in_ch2 + self.out_ch2
        self.out_ch3 = out_ch * 6
        self.conv3 = nn.Sequential(
            nn.Conv3d(self.in_ch3, self.out_ch3, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(self.out_ch3),
            nn.ReLU(inplace=True)
        )
        self.in_ch4 = self.in_ch3 + self.out_ch3
        self.bottleneck = nn.Sequential(
            nn.Conv3d(self.in_ch4, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_ch),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        x1 = torch.cat([out1, x], dim=1)
        out2 = self.conv2(x1)
        x2 = torch.cat([out2, x1], dim=1)
        out3 = self.conv3(x2)
        x3 = torch.cat([out3, x2], dim=1)
        output = self.bottleneck(x3)

        return output

