import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(18, 256, 5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 18, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = torch.empty_like(x).copy_(x)
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 4, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(8*8*4, 256)
        self.fc2 = nn.Linear(256, 1)

        self.conv1 = nn.Conv2d(256, 2, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8*8*2, 1968)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, 4*8*8)
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, 2*8*8)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(7):
            setattr(self, f"res_{block}", ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(7):
            s = getattr(self, f"res_{block}")(s)
        return self.outblock(s)
