import torch.nn as nn
import torch.nn.functional as F
import torch


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SEModule3d(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class Res2Block3d(nn.Module):
    """
    The Res2Net Block
    """

    def __init__(self, in_channels, out_channels, scales=4, stride=1, expansion=4):
        super(Res2Block3d, self).__init__()
        if out_channels % scales != 0:
            raise ValueError(
                "channels must be divisible by scales. out_channels:{0:d}, scales:{1:d}".format(out_channels, scales))
        self.sub_channels = out_channels // scales
        self.scales = scales
        self.expansion = expansion
        self.stride = stride
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=self.stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        if scales == 1:
            self.nums = 1
        else:
            self.nums = scales - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv3d(self.sub_channels, self.sub_channels, kernel_size=3, stride=self.stride, padding=1, bias=False))
            bns.append(nn.BatchNorm3d(self.sub_channels))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv3d(self.sub_channels * scales, self.sub_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        self.se = SEModule3d(self.sub_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv1x1(x)

        out = self.relu(self.bn1(self.conv1(x)))
        # We use torch.split operation to split the features into sub-block features in channels wise
        split_x = torch.split(out, self.sub_channels, dim=1)
        for i in range(self.nums):
            if i == 0:
                split = split_x[i]
            else:
                split = split + split_x[i]
            split = self.relu(self.bns[i](self.convs[i](split)))
            if i == 0:
                out = split
            else:
                out = torch.cat((out, split), dim=1)

        if self.scales > 1:
            out = torch.cat((out, split_x[self.nums]), dim=1)

        out = self.relu(self.conv3(out))
        out = self.se(out)
        out = out + residual
        out = self.relu(out)
        return out


class UNetBasicBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBasicBlock3d, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm3d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm3d(out_channels),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class UNet3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3d, self).__init__()
        self.enc1 = UNetBasicBlock3d(in_channels, 16)
        self.enc2 = UNetBasicBlock3d(16, 16)
        self.enc3 = UNetBasicBlock3d(16, 16)
        self.bridge = UNetBasicBlock3d(16, 16)
        self.dec3 = UNetBasicBlock3d(32, 16)
        self.dec2 = UNetBasicBlock3d(32, 16)
        self.dec1 = UNetBasicBlock3d(32, 16)
        self.down = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.up3 = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.finalconv = nn.Conv3d(16, out_channels, kernel_size=1)
        initialize_weights(self)

    def forward(self, x, coarse_map):
        resudual = x
        enc1 = self.enc1(coarse_map)
        down1 = self.down(enc1)
        enc2 = self.enc2(down1)
        down2 = self.down(enc2)
        enc3 = self.enc3(down2)
        down3 = self.down(enc3)
        bridge = self.bridge(down3)
        up3 = self.up3(bridge)
        up3 = torch.cat((up3, enc3), 1)
        dec3 = self.dec3(up3)
        up2 = self.up2(dec3)
        up2 = torch.cat((up2, enc2), 1)
        dec2 = self.dec2(up2)
        up1 = self.up1(dec2)
        up1 = torch.cat((up1, enc1), 1)
        dec1 = self.dec1(up1)
        out = self.finalconv(dec1)
        out = resudual + out
        return out


class SpatialAttentionBlock3d(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SpatialAttentionBlock3d, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.key = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.judge = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxZ )
        :return: affinity value + x
        B: batch size
        C: channels
        H: height
        W: width
        D: volume number (depth)
        """
        B, C, H, W, D = x.size()
        # compress x: [B,C,H,W,Z]-->[B,H*W*Z,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,W*H*D,C]
        proj_key = self.key(x).view(B, -1, W * H * D)  # -> [B,H*W*D,C]
        proj_judge = self.judge(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,C,H*W*D]

        affinity1 = torch.matmul(proj_query, proj_key)  # -> [B,W*H,W*H]
        affinity2 = torch.matmul(proj_judge, proj_key)  # -> [B,W*H,W*H]
        affinity = torch.matmul(affinity1, affinity2)  # -> [B,W*H,W*H]
        affinity = self.softmax(affinity)

        proj_value = self.value(x).view(B, -1, H * W * D)  # -> C*N
        weights = torch.matmul(proj_value, affinity)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        # out = weights + x
        return out


class ChannelAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock3d, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W, D = x.size()
        proj_query = x.view(B, C, -1).permute(0, 2, 1)
        proj_key = x.view(B, C, -1)
        proj_judge = x.view(B, C, -1).permute(0, 2, 1)
        affinity1 = torch.matmul(proj_key, proj_query)
        affinity2 = torch.matmul(proj_key, proj_judge)
        affinity = torch.matmul(affinity1, affinity2)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out


class AffinityAttention3d(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention3d, self).__init__()
        self.sab = SpatialAttentionBlock3d(in_channels)
        self.cab = ChannelAttentionBlock3d(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab + x
        return out


class ResEncoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out += residual
        out = self.relu(out)
        return out


class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class FinerRes2CSNet(nn.Module):
    def __init__(self, in_channels, classes, scales=4, expansion=4):
        super(FinerRes2CSNet, self).__init__()
        if scales != expansion:
            raise ValueError("scales and expansion must be the same")
        self.encoder1 = Res2Block3d(in_channels, 64, scales=scales, expansion=expansion)
        self.encoder2 = Res2Block3d(64, 128, scales=scales, expansion=expansion)
        self.encoder3 = Res2Block3d(128, 256, scales=scales, expansion=expansion)
        self.bridge = Res2Block3d(256, 512)
        self.down = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.decoder3 = Res2Block3d(512, 256, scales=scales, expansion=expansion)
        self.decoder2 = Res2Block3d(256, 128, scales=scales, expansion=expansion)
        self.decoder1 = Res2Block3d(128, 64, scales=scales, expansion=expansion)
        self.upsample3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.final = nn.Conv3d(64, classes, kernel_size=1, stride=1, bias=False)
        self.refine = UNet3d(classes + 1, classes)
        initialize_weights(self)

    def forward(self, x):
        # Encoder Path
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)
        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)
        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)
        bridge = self.bridge(down3)
        up3 = self.upsample3(bridge)
        up3 = torch.cat((up3, enc3), dim=1)
        dec3 = self.decoder3(up3)
        up2 = self.upsample2(dec3)
        up2 = torch.cat((up2, enc2), dim=1)
        dec2 = self.decoder2(up2)
        up1 = self.upsample1(dec2)
        up1 = torch.cat((up1, enc1), dim=1)
        dec1 = self.decoder1(up1)

        # Final output
        coarse_map = self.final(dec1)
        coarse_input = torch.cat((coarse_map, x), dim=1)
        coarse_input = F.softmax(coarse_input, dim=1)
        fine_map = self.refine(coarse_map, coarse_input)

        return coarse_map, fine_map
