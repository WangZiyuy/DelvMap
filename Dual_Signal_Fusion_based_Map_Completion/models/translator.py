from .basic_block import *
import torch

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.backends.cudnn.benchmark = False


class Unet(nn.Module):

    def __init__(self, input_nc, conv1d=False):
        super(Unet, self).__init__()

        self.down1 = self.conv_stage(input_nc, 64)
        self.down2 = self.conv_stage(64, 128)
        self.down3 = self.conv_stage(128, 256)
        self.down4 = self.conv_stage(256, 512)

        self.center = self.conv_stage(512, 1024)

        self.up4 = self.conv_stage(1024, 512)
        self.up3 = self.conv_stage(512, 256)
        self.up2 = self.conv_stage(256, 128)
        self.up1 = self.conv_stage(128, 64)

        if conv1d:
            self.upsample = DecoderBlock1DConv4
        else:
            self.upsample = self.upsample_original
        self.trans4 = self.upsample(1024, 512)
        self.trans3 = self.upsample(512, 256)
        self.trans2 = self.upsample(256, 128)
        self.trans1 = self.upsample(128, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.max_pool = nn.MaxPool2d(2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    # m.weight.data.normal_(mean=0.0, std=1.0)
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),

            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )

    def upsample_original(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        conv1_out = self.down1(x)
        conv2_out = self.down2(self.max_pool(conv1_out))
        conv3_out = self.down3(self.max_pool(conv2_out))
        conv4_out = self.down4(self.max_pool(conv3_out))

        out = self.center(self.max_pool(conv4_out))

        out = self.up4(torch.cat((self.trans4(out), conv4_out), 1))
        out = self.up3(torch.cat((self.trans3(out), conv3_out), 1))
        out = self.up2(torch.cat((self.trans2(out), conv2_out), 1))
        out = self.up1(torch.cat((self.trans1(out), conv1_out), 1))

        out = self.conv_last(out)
        return out


# final model
class DSFNet(nn.Module):

    def __init__(self, input_nc, conv1d=False):
        super(DSFNet, self).__init__()
        # buiding 编解码
        self.down1_src_building = self.conv_stage(3, 64)
        self.down2_src_building = self.conv_stage(64, 128)
        self.down3_src_building = self.conv_stage(128, 256)
        self.down4_src_building = self.conv_stage(256, 512)

        self.center_src_building = self.conv_stage(512, 1024)

        self.up4_src_building = self.conv_stage(1024, 512)
        self.up3_src_building = self.conv_stage(512, 256)
        self.up2_src_building = self.conv_stage(256, 128)
        self.up1_src_building = self.conv_stage(128, 64)

        # 定义STFuse中gate conv stage
        self.gate1 = self.gate_conv_stage(2048, 2)
        self.gate2 = self.gate_conv_stage(1024, 2)
        self.gate3 = self.gate_conv_stage(512, 2)
        self.gate4 = self.gate_conv_stage(256, 2)
        self.gate5 = self.gate_conv_stage(128, 2)

        if conv1d:
            self.upsample = DecoderBlock1DConv4
        else:
            self.upsample = self.upsample_original

        self.trans4_src_building = self.upsample(1024, 512)
        self.trans3_src_building = self.upsample(512, 256)
        self.trans2_src_building = self.upsample(256, 128)
        self.trans1_src_building = self.upsample(128, 64)

        # traj特征编解码
        self.down1_traj = self.conv_stage(2, 64)
        self.down2_traj = self.conv_stage(64, 128)
        self.down3_traj = self.conv_stage(128, 256)
        self.down4_traj = self.conv_stage(256, 512)

        self.center_traj = self.conv_stage(512, 1024)

        self.up4_traj = self.conv_stage(1024, 512)
        self.up3_traj = self.conv_stage(512, 256)
        self.up2_traj = self.conv_stage(256, 128)
        self.up1_traj = self.conv_stage(128, 64)

        if conv1d:
            self.upsample = DecoderBlock1DConv4
        else:
            self.upsample = self.upsample_original

        self.trans4_traj = self.upsample(1024, 512)
        self.trans3_traj = self.upsample(512, 256)
        self.trans2_traj = self.upsample(256, 128)
        self.trans1_traj = self.upsample(128, 64)

        # 遥感轨迹特征编解码
        self.down1_src_traj = self.conv_stage(3, 64)
        self.down2_src_traj = self.conv_stage(64, 128)
        self.down3_src_traj = self.conv_stage(128, 256)
        self.down4_src_traj = self.conv_stage(256, 512)

        self.center_src_traj = self.conv_stage(512, 1024)

        self.up4_src_traj = self.conv_stage(1024, 512)
        self.up3_src_traj = self.conv_stage(512, 256)
        self.up2_src_traj = self.conv_stage(256, 128)
        self.up1_src_traj = self.conv_stage(128, 64)

        if conv1d:
            self.upsample = DecoderBlock1DConv4
        else:
            self.upsample = self.upsample_original

        self.trans4_src_traj = self.upsample(1024, 512)
        self.trans3_src_traj = self.upsample(512, 256)
        self.trans2_src_traj = self.upsample(256, 128)
        self.trans1_src_traj = self.upsample(128, 64)

        if conv1d:
            self.upsample = DecoderBlock1DConv4
        else:
            self.upsample = self.upsample_original

        # traj主任务头
        self.conv_last = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        # building 任务头
        self.conv_last1 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        # 遥感影像提取路网任务头
        self.conv_last2 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    # m.weight.data.normal_(mean=0.0, std=1.0)
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    m.bias.data.zero_()

        self.sfw1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sfw2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sfw3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sfw4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sfw1.data.fill_(0.25)
        self.sfw2.data.fill_(0.25)
        self.sfw3.data.fill_(0.25)
        self.sfw4.data.fill_(0.25)

        self.gate_g1_up = self.ca_map_upsampling()
        self.gate_g2_up = self.ca_map_upsampling()
        self.gate_g3_up = self.ca_map_upsampling()
        self.gate_g4_up = self.ca_map_upsampling()
        self.gate_g5_up = self.ca_map_upsampling()

        # 定义STFuse中co attention的融合特征图缩减通道
        self.ca_info_down1 = self.ca_conv_stage(2048, 1024)
        self.ca_info_down2 = self.ca_conv_stage(1024, 512)
        self.ca_info_down3 = self.ca_conv_stage(512, 256)
        self.ca_info_down4 = self.ca_conv_stage(256, 128)
        self.ca_info_down5 = self.ca_conv_stage(128, 64)

        self.ca_soft_down1 = self.ca_conv_stage(256, 2)
        self.ca_soft_down2 = self.ca_conv_stage(1024, 2)
        self.ca_soft_down3 = self.ca_conv_stage(4096, 2)
        self.ca_soft_down4 = self.ca_conv_stage(16384, 2)

        # 用于计算BTFuse输出
        self.temp = torch.ones((256, 256), device=torch.device("cuda"))

        # 用于STFuse关联矩阵增加权重
        self.W_b = nn.Parameter(torch.ones(1024, 1024))
        self.W_s = nn.Parameter(torch.ones(256, 1024))
        self.W_t = nn.Parameter(torch.ones(256, 1024))

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),

            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )

    def gate_conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
        )

    def ca_map_upsampling(self):
        return nn.Sequential(
            nn.Upsample(scale_factor=2)
        )

    def ca_conv_stage(self, dim_in, dim_out, kernel_size=1, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            # nn.BatchNorm2d(dim_out),
            # nn.ReLU()
        )

    def upsample_original(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU()
        )


    def co_att_first(self, feature1, feature2):
        fs1, ft1 = feature1, feature2
        B, N, W, H = fs1.shape
        info = torch.cat((fs1, ft1), 1)  # 除了权重还应生成一个特征图 [1, 2048, 16, 16]
        info = self.ca_info_down1(info)

        x1 = fs1.reshape(B, N, W * H)  # [1, 1024, 256]
        x2 = ft1.reshape(B, N, W * H)
        C = torch.bmm(x2.permute(0, 2, 1), torch.matmul(self.W_b, x1))  # （256，1024) * ((1024,1024)*(1024,256))
        Hs = nn.Tanh()(torch.matmul(self.W_s, x1) + torch.matmul(torch.matmul(self.W_t, x2), C))

        WI = F.softmax(self.ca_soft_down1(Hs.unsqueeze(-1)), dim=1).squeeze(-1)  # [1, 2, 256]
        WI1, WI2 = WI[:, 0, :].unsqueeze(1), WI[:, 1, :].unsqueeze(1)
        fs1_prime = (x1 * WI1 + x2 * WI2).reshape(B, N, W, H)  # [1, 1024, 256] [1, 1024, 16,16]

        return info, fs1_prime

    def co_att(self, feature1, feature2, wb, wt, ws, ca_soft_down):
        fs, ft = feature1, feature2
        B, N, W, H = fs.shape
        info = torch.cat((fs, ft), 1)  # 除了权重还应生成一个特征图 [1, 2048, 16, 16]

        x1 = fs.reshape(B, N, W * H)  # [1, 1024, 256]
        x2 = ft.reshape(B, N, W * H)
        C = torch.bmm(x2.permute(0, 2, 1), torch.matmul(wb, x1))  # （256，1024) * ((1024,1024)*(1024,256))
        Hs = nn.Tanh()(torch.matmul(ws, x1) + torch.matmul(torch.matmul(wt, x2), C))

        WI = F.softmax(ca_soft_down(Hs.unsqueeze(-1)), dim=1).squeeze(-1)  # [1, 2, 256]
        WI1, WI2 = WI[:, 0, :].unsqueeze(1), WI[:, 1, :].unsqueeze(1)
        fs_prime = (x1 * WI1 + x2 * WI2).reshape(B, N, W, H)  # [1, 1024, 256] [1, 1024, 16,16]

        return info, fs_prime

    def forward(self, traj, src):
        """
        src:遥感图像
        traj:轨迹特征
        """
        # 遥感图像建筑提取编码器(SB)
        sb_conv1_out = self.down1_src_building(src)
        sb_conv2_out = self.down2_src_building(self.max_pool(sb_conv1_out))
        sb_conv3_out = self.down3_src_building(self.max_pool(sb_conv2_out))
        sb_conv4_out = self.down4_src_building(self.max_pool(sb_conv3_out))
        sb_center_out = self.center_src_building(self.max_pool(sb_conv4_out))

        sb_out4 = self.up4_src_building(torch.cat((self.trans4_src_building(sb_center_out), sb_conv4_out), 1))  # sb_out 1024  sb_conv4_out 512
        sb_out3 = self.up3_src_building(torch.cat((self.trans3_src_building(sb_out4), sb_conv3_out), 1))
        sb_out2 = self.up2_src_building(torch.cat((self.trans2_src_building(sb_out3), sb_conv2_out), 1))
        sb_out1 = self.up1_src_building(torch.cat((self.trans1_src_building(sb_out2), sb_conv1_out), 1))
        sb_out = self.conv_last1(sb_out1)

        max_channel = torch.max(sb_out1, dim=2, keepdim=True)[0]
        max_channel = torch.max(max_channel, dim=3, keepdim=True)[0] # [1,64,1,1]
        temp = torch.ones(256, 256).cuda()
        btfuse_out = (max_channel * temp - sb_out1) / (max_channel + 1e-8)

        tr_conv1_out = self.down1_traj(traj)
        tr_conv2_out = self.down2_traj(self.max_pool(tr_conv1_out * btfuse_out)) # 256,256,64
        tr_conv3_out = self.down3_traj(self.max_pool(tr_conv2_out))
        tr_conv4_out = self.down4_traj(self.max_pool(tr_conv3_out))
        tr_center_out = self.center_traj(self.max_pool(tr_conv4_out))

        tr_out4 = self.up4_traj(torch.cat((self.trans4_traj(tr_center_out), tr_conv4_out), 1))
        tr_out3 = self.up3_traj(torch.cat((self.trans3_traj(tr_out4), tr_conv3_out), 1))
        tr_out2 = self.up2_traj(torch.cat((self.trans2_traj(tr_out3), tr_conv2_out), 1))
        tr_out1 = self.up1_traj(torch.cat((self.trans1_traj(tr_out2), tr_conv1_out), 1))
        tr_out = self.conv_last(tr_out1)

        # 遥感图像道路提取编码器(SR)
        sr_conv1_out = self.down1_src_traj(src)
        sr_conv2_out = self.down2_src_traj(self.max_pool(sr_conv1_out))
        sr_conv3_out = self.down3_src_traj(self.max_pool(sr_conv2_out))
        sr_conv4_out = self.down4_src_traj(self.max_pool(sr_conv3_out))
        sr_center_out = self.center_src_traj(self.max_pool(sr_conv4_out))

        fi1, ft1 = sr_center_out, tr_center_out  # [4, 1024, 16, 16]
        info1, fi_ca1 = self.co_att_first(fi1, ft1)
        sr_out4 = self.up4_src_traj(torch.cat((self.trans4_src_traj(fi_ca1), sr_conv4_out), 1))
        sr_out3 = self.up3_src_traj(torch.cat((self.trans3_src_traj(sr_out4), sr_conv3_out), 1))
        sr_out2 = self.up2_src_traj(torch.cat((self.trans2_src_traj(sr_out3), sr_conv2_out), 1))
        sr_out1 = self.up1_src_traj(torch.cat((self.trans1_src_traj(sr_out2), sr_conv1_out), 1))
        sr_out = self.conv_last2(sr_out1)

        return tr_out, sb_out, sr_out

