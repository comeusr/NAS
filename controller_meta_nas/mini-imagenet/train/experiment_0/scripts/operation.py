from metalayer import *
from torchviz import make_dot

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: Conv_7x1_1x7(C, stride, affine),
}


class Conv_7x1_1x7(nn.Module):

    def __init__(self, C, stride, affine):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = MetaConv2dLayer(C, C, (1, 7), stride=(1, stride), padding=(0, 3), use_bias=False)
        self.conv2 = MetaConv2dLayer(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), use_bias=False)
        self.bn = nn.BatchNorm2d(C, affine=affine)

    def foward(self, x, params=None):
        if params is not None:
            param_conv1 = params['conv1']
            param_conv2 = params['conv2']

        out = self.relu(x)
        out = self.conv1(out, param_conv1)
        out = self.conv2(out, param_conv2)
        out = self.bn(out)
        return out


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
        super().__init__()

        self.conv = MetaConv2dLayer(C_in, C_out, kernel_size, stride=stride, padding=padding, use_bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.relu = nn.ReLU(inplace=False)

        # self.op = nn.Sequential(
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
        #     nn.BatchNorm2d(C_out, affine=affine)
        # )

    def forward(self, x, params=None):
        conv_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            conv_params = params['conv']

        out = self.relu(x)
        out = self.conv(out, conv_params)
        out = self.bn(out)
        return out


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()

        self.relu = nn.ReLU(inplace=False)
        self.conv1 = MetaConv2dLayer(1, C_in, kernel_size, stride, padding, use_bias=False, groups=C_in, dilation_rate=dilation)
        self.conv2 = MetaConv2dLayer(C_in, C_out, kernel_size=1, stride=1, padding=0, use_bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

        # self.op = nn.Sequential(
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
        #               groups=C_in, bias=False),
        #     nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(C_out, affine=affine),
        # )

    def forward(self, x, params):
        conv1_params = None
        conv2_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            conv1_params = params['conv1']
            conv2_params = params['conv2']

        out = self.relu(x)
        out = self.conv1(out, params=conv1_params)
        out = self.conv2(out, params=conv2_params)
        out = self.bn(out)
        return out


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()

        self.relu = nn.ReLU(inplace=False)
        self.conv1 = MetaConv2dLayer(1, C_in, kernel_size, stride, padding, use_bias=False, groups=C_in)
        self.conv2 = MetaConv2dLayer(C_in, C_in, kernel_size=1, stride=1, padding=0, use_bias=False)
        self.bn1 = nn.BatchNorm2d(C_in, affine=affine)
        self.conv3 = MetaConv2dLayer(1, C_in, kernel_size, stride=1, padding=padding, use_bias=False, groups=C_in)
        self.conv4 = MetaConv2dLayer(C_in, C_out, kernel_size=1, stride=1, padding=0, use_bias=False)
        self.bn2 = nn.BatchNorm2d(C_out, affine=affine)

        # self.op = nn.Sequential(
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
        #     nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(C_in, affine=affine),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
        #     nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(C_out, affine=affine),
        # )

    def forward(self, x, params):
        conv1_params = None
        conv2_params = None
        conv3_params = None
        conv4_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            conv1_params = params["conv1"]
            conv2_params = params["conv2"]
            conv3_params = params["conv3"]
            conv4_params = params["conv4"]

        out = self.relu(x)
        out = self.conv1(out, conv1_params)
        out = self.conv2(out, conv2_params)
        out = self.bn1(out)
        out = self.conv3(out, conv3_params)
        out = self.conv4(out, conv4_params)
        out = self.bn2(out)
        return out


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class stem0(nn.Module):

    def __init__(self, C_curr, kernel_size, stride, padding, bias, device, args):
        super().__init__()
        # nn.Conv2d(3, C_curr // 2, kernel_size=3, padding=1, bias=False),
        #       nn.BatchNorm2d(C_curr // 2),
        #       nn.MaxPool2d(2, 2),
        #       nn.ReLU(inplace=True),
        #       nn.Conv2d(C_curr // 2, C_curr, kernel_size=3, padding=1, bias=False),
        #       nn.BatchNorm2d(C_curr),
        #       nn.MaxPool2d(2, 2),

        self.conv1 = MetaConv2dLayer(3, C_curr // 2, kernel_size=kernel_size, stride=stride, padding=padding, use_bias=bias)
        self.bn1 = MetaBatchNormLayer(C_curr // 2, device, args)  # nn.BatchNorm2d(C_curr//2)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MetaConv2dLayer(C_curr // 2, C_curr, kernel_size=kernel_size, stride=stride, padding=padding, use_bias=bias)
        self.bn2 = MetaBatchNormLayer(C_curr, device, args)  # nn.BatchNorm2d(C_curr)
        self.maxpool2 = nn.MaxPool2d(2, 2)

    def forward(self, x, params, step, training=True, backup_running_statistics=False):
        if params is not None:
            params_dict = extract_top_level_dict(params)
            params_conv1 = params_dict['conv1']
            params_conv2 = params_dict['conv2']
            params_bn1 = params_dict['bn1']
            params_bn2 = params_dict['bn2']

        out = self.conv1(x, params_conv1)
        out = self.bn1(out, num_step=step, params=params_bn1, training=training,
                       backup_running_statistics=backup_running_statistics)
        out = self.maxpool1(out)
        out = self.relu(out)
        out = self.conv2(out, params_conv2)
        out = self.bn2(out, num_step=step, params=params_bn2, training=training,
                       backup_running_statistics=backup_running_statistics)
        out = self.maxpool2(out)

        # make_dot(out, params).render('bn', format='pdf')
        return out


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        # self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        # self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

        self.conv1 = MetaConv2dLayer(C_in, C_out // 2, kernel_size=1, stride=2, padding=0, use_bias=False)
        self.conv2 = MetaConv2dLayer(C_in, C_out // 2, kernel_size=1, stride=2, padding=0, use_bias=False)

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(params)
            param_conv1 = params['conv1']
            param_conv2 = params['conv2']

        x = self.relu(x)
        out = torch.cat([self.conv1(x, param_conv1), self.conv2(x, param_conv2)], dim=1)
        out = self.bn(out)
        return out
