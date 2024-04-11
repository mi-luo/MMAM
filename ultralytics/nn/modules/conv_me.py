import torch
import torch.nn as nn


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))



class ShiftConv(nn.Module):
    def __init__(self, c1, c2, k=1):
        super().__init__()
        self.conv1x1 = nn.Conv2d(c1, c2, k)
        self.conv2 = nn.Conv2d(c1, c2, k=3, s=1, p=1)
        self.cv1 = nn.Conv2d(2,1,k=3,s=1,bias=False)
        self.act = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    def spatial_shift(self,x):
        b,w,h,c = x.size()
        x_clone = x.clone()
        x[:,1:,:,:c/4] = x_clone[:,:w-1,:,:c/4]  # 注 c一定要是4的整数倍！
        x[:,:w-1,:,c/4:c/2] = x_clone[:,1:,:,c/4:c/2]
        x[:,:,1:,c/2:c*3/4] = x_clone[:,:,:h-1,c/2:c*3/4]
        x[:,:,:h-1,3*c/4:] = x_clone[:,:,1:,3*c/4:]
        return x

    def forward(self, x):
        x = self.spatial_shift(x) # 空间shift操作
        x = self.act(self.bn(self.conv2(x))) # 卷积+ BN +激活
        x = x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1))) # 空间注意力
        x = self.conv1x1(x) # 调节通道用的 暂时还没修改
        return x


class GhostConv1(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)
        self.cv3 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x) # 卷积
        y1 = torch.cat((y, self.cv2(y)), 1) # 两部分通道卷积拼接
        y = y1 * self.act(self.cv3(torch.cat([torch.mean(y1, 1, keepdim=True), torch.max(y1, 1, keepdim=True)[0]], 1)))
        return y