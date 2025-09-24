import torch
import torch.nn as nn


class CSPliteConv_maxpool(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super().__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))

class CSPliteShuffleNetV2(nn.Module):
    def __init__(self, inp, oup, stride):  # ch_in, ch_out, stride
        super().__init__()

        self.stride = stride  #決定特徵圖的尺寸 （==1爲完整特徵圖， ==2爲一半尺寸的特徵圖）

        branch_features = oup // 2
        branch_features1 = oup // 4
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride == 2: # for 下採樣
            # copy input
            self.branch1 = nn.Sequential(  # left
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp),
                nn.BatchNorm2d(inp),
                
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()
          # CSP 分支
        self.csp_conv = nn.Sequential(
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(  # right01(for down)
            nn.Conv2d(inp , branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            #如果stride=2，其通道數爲inp，反之爲branch _ feature
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),

            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, groups=branch_features),
            nn.BatchNorm2d(branch_features),

            # nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),  # for liteconv 改進
            # nn.BatchNorm2d(branch_features),
            # nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(  # right02 (for basic)
            nn.Conv2d(branch_features1 , branch_features1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features1),
            nn.ReLU(inplace=True),

            nn.Conv2d(branch_features1, branch_features1, kernel_size=3, stride=self.stride, padding=1, groups=branch_features1),
            nn.BatchNorm2d(branch_features1),

            # nn.Conv2d(branch_features1, branch_features1, kernel_size=1, stride=1, padding=0, bias=False),  # for liteconv 改進
            # nn.BatchNorm2d(branch_features1),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
       if self.stride == 1:
           direct_path, processed_path = x.chunk(2, dim=1) # 將輸入分為兩半
           x1, x2 = processed_path.chunk(2, dim=1)  # 將 processed_path 再次分為兩半
           basic_out = torch.cat((x1, self.branch3(x2)), dim=1) # x3 直接保留，x4 進入分支網絡
           csp_out = self.csp_conv(direct_path)# direct_path 進入 CSP 分支
           out = torch.cat((csp_out, basic_out), dim=1)# 拼接 direct_path 和 basic_out
       else:
           # stride=2 的情況
          out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
      
       out = self.channel_shuffle(out, 2)

       return out

    def channel_shuffle(self, x, groups):
        N, C, H, W = x.size()
        out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

        return out
