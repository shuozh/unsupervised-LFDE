import torch
import torch.nn as nn
from einops import rearrange, repeat

class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(fn, fn, 3, padding=1)
        self.conv2 = nn.Conv2d(fn, fn, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(fn)
        self.norm2 = nn.BatchNorm2d(fn)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        return identity + out
    
class UNetRGB(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_ch = 49*3
        self.ch1 = nn.Conv2d(in_ch, 64, 3, 1, 1)
        self.conv1 = self.make_layer(ResidualBlock, 64, 2)
        self.pool1 = nn.MaxPool2d(2)

        self.ch2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2 = self.make_layer(ResidualBlock, 128, 2)
        self.pool2 = nn.MaxPool2d(2)

        self.ch3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3 = self.make_layer(ResidualBlock, 256, 2)
        self.pool3 = nn.MaxPool2d(2)

        self.ch4 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4 = self.make_layer(ResidualBlock, 512, 2)


        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ch7 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv7 = self.make_layer(ResidualBlock, 256, 2)
        # self.head7_d = nn.Conv2d(256, 1, 1)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ch8 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv8 = self.make_layer(ResidualBlock, 128, 2)
        # self.head8_d = nn.Conv2d(128, 1, 1)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ch9 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv9 = self.make_layer(ResidualBlock, 64, 2)
        self.head9_d = nn.Conv2d(64, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.index_49 = [
            10, 11, 12, 13, 14, 15, 16,
            19, 20, 21, 22, 23, 24, 25, 
            28, 29, 30, 31, 32, 33, 34,
            37, 38, 39, 40, 41, 42, 43,
            46, 47, 48, 49, 50, 51, 52,
            55, 56, 57, 58, 59, 60, 61,
            64, 65, 66, 67, 68, 69, 70, 
        ]
        # self.center_index = [30, 31, 32, 39, 40, 41, 48, 49, 50]
        self.tanh = nn.Tanh()
    def make_layer(self, block, p, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block(p))
        return nn.Sequential(*layers)

    def forward(self, x):
        xraw = rearrange(x, 'b u v h w c -> b (u v c) h w')
        c1 = self.conv1(self.relu(self.ch1(xraw)))
        p1 = self.pool1(c1)
        c2 = self.conv2(self.relu(self.ch2(p1)))
        p2 = self.pool2(c2)
        c3 = self.conv3(self.relu(self.ch3(p2)))
        p3 = self.pool3(c3)
        c4 = self.conv4(self.relu(self.ch4(p3)))
        
        up_7 = self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(self.relu(self.ch7(merge7)))
        # disp7 = self.head7_d(c7)
        # out7 = disp7

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(self.relu(self.ch8(merge8)))
        # disp8 = self.head8_d(c8)
        # out8 = disp8

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(self.relu(self.ch9(merge9)))
        disp9 = self.tanh(self.head9_d(c9))
        disp9 = disp9 * 4
        return disp9.squeeze(1)

class UNetRGBReal(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_ch = 49*3
        self.ch1 = nn.Conv2d(in_ch, 64, 3, 1, 1)
        self.conv1 = self.make_layer(ResidualBlock, 64, 2)
        self.pool1 = nn.MaxPool2d(2)

        self.ch2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2 = self.make_layer(ResidualBlock, 128, 2)
        self.pool2 = nn.MaxPool2d(2)

        self.ch3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3 = self.make_layer(ResidualBlock, 256, 2)
        self.pool3 = nn.MaxPool2d(2)

        self.ch4 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4 = self.make_layer(ResidualBlock, 512, 2)


        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ch7 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv7 = self.make_layer(ResidualBlock, 256, 2)
        # self.head7_d = nn.Conv2d(256, 1, 1)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ch8 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv8 = self.make_layer(ResidualBlock, 128, 2)
        # self.head8_d = nn.Conv2d(128, 1, 1)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ch9 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv9 = self.make_layer(ResidualBlock, 64, 2)
        self.head9_d = nn.Conv2d(64, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        # self.center_index = [30, 31, 32, 39, 40, 41, 48, 49, 50]
        self.tanh = nn.Tanh()
    def make_layer(self, block, p, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block(p))
        return nn.Sequential(*layers)

    def forward(self, x):
        xraw = rearrange(x, 'b u v h w c -> b (u v c) h w')
        c1 = self.conv1(self.relu(self.ch1(xraw)))
        p1 = self.pool1(c1)
        c2 = self.conv2(self.relu(self.ch2(p1)))
        p2 = self.pool2(c2)
        c3 = self.conv3(self.relu(self.ch3(p2)))
        p3 = self.pool3(c3)
        c4 = self.conv4(self.relu(self.ch4(p3)))
        
        up_7 = self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(self.relu(self.ch7(merge7)))
        # disp7 = self.head7_d(c7)
        # out7 = disp7

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(self.relu(self.ch8(merge8)))
        # disp8 = self.head8_d(c8)
        # out8 = disp8

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(self.relu(self.ch9(merge9)))
        disp9 = self.tanh(self.head9_d(c9))
        disp9 = disp9 * 2
        return disp9.squeeze(1)
    
class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, padding='same', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class EPINet(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.index_49 = [
            10, 11, 12, 13, 14, 15, 16,
            19, 20, 21, 22, 23, 24, 25, 
            28, 29, 30, 31, 32, 33, 34,
            37, 38, 39, 40, 41, 42, 43,
            46, 47, 48, 49, 50, 51, 52,
            55, 56, 57, 58, 59, 60, 61,
            64, 65, 66, 67, 68, 69, 70, 
        ]
        self.center_index = [30, 31, 32, 39, 40, 41, 48, 49, 50]
        self.branch_index = [
            [21, 22, 23, 24, 25, 26, 27],
            [3, 10, 17, 24, 31, 48, 45],
            [0, 8, 16, 24, 32, 40, 48],
            [6, 12, 18, 24, 30, 36, 42]
        ]
        self.f0 = nn.Sequential(
            BasicBlock(7*3, 70),
            BasicBlock(70, 70),
            BasicBlock(70, 70)
        )
        self.f1 = nn.Sequential(
            BasicBlock(7*3, 70),
            BasicBlock(70, 70),
            BasicBlock(70, 70)
        )
        self.f2 = nn.Sequential(
            BasicBlock(7*3, 70),
            BasicBlock(70, 70),
            BasicBlock(70, 70)
        )
        self.f3 = nn.Sequential(
            BasicBlock(7*3, 70),
            BasicBlock(70, 70),
            BasicBlock(70, 70)
        )
        self.d = nn.Sequential(
            BasicBlock(280, 280),
            BasicBlock(280, 280),
            BasicBlock(280, 280),
            BasicBlock(280, 280),
            BasicBlock(280, 280),
            BasicBlock(280, 280),
            BasicBlock(280, 280),
            nn.Conv2d(280, 280, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(280, 1, 3, padding='same')
        )

    def forward(self, x):
        # i = random.randint(0, 8)
        # i = 4
        # bias = self.center_index[i] - 40
        # index49 = [a+bias for a in self.index_49]
        # x = x[:, index49, :, :]
        # xraw = x
        x = rearrange(x, 'b u v h w c -> b (u v) c h w')
        x0 = self.f0(rearrange(x[:, self.branch_index[0], ...], 'b n c h w -> b (n c) h w'))
        x1 = self.f0(rearrange(x[:, self.branch_index[1], ...], 'b n c h w -> b (n c) h w'))
        x2 = self.f0(rearrange(x[:, self.branch_index[2], ...], 'b n c h w -> b (n c) h w'))
        x3 = self.f0(rearrange(x[:, self.branch_index[3], ...], 'b n c h w -> b (n c) h w'))
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.d(x)
        x = torch.squeeze(x, dim=1)
        # out = [xraw, x]
        return x
        # return out

