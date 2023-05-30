import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class xxNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(xxNet, self).__init__()
        # 初始化一些参数(opt)
        self.input_channels = input_channels
        self.output_channels = output_channels

        # 网络中需要的层(可单独写，如果大量的重复可以封装成Sequential变成一个模块)
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 5, 1, 2),
            nn.MaxPool2d(2),
        )

    # 前向传播
    def forward(self, x):
        x = self.model(x)
        return x


# 创建一个实例去验证一下网络模型搭建是否有问题，同时也可以去反向计算一些层的参数是什么，例如linear层
if __name__ == '__main__':
    writer = SummaryWriter("../logs/model_logs")

    net = xxNet(input_channels=1, output_channels=2)
    print(net)

    input = torch.ones((1, 3, 32, 32))
    output = net(input)
    print(output.shape)

    writer.add_graph(net, input)
    writer.close()