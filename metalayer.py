import numbers
from copy import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from collections import OrderedDict


def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = {}
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level in output_dict:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

        elif sub_level == "":
            output_dict[top_level] = current_dict[key]
        else:
            output_dict[top_level] = {sub_level: current_dict[key]}
    # print(current_dict.keys(), output_dict.keys())
    return OrderedDict(output_dict)


class MetaConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, groups=1, dilation_rate=1):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.groups = int(groups)
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        # print('Self Weight shape {}'.format(self.weight.shape))

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        elif self.use_bias:
            weight, bias = self.weight, self.bias
        else:
            weight = self.weight
            bias = None

        return F.conv2d(
            input=x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation_rate,
            groups=self.groups,
        )


class MetaLinearLayer(nn.Module):
    def __init__(self, in_shape, out_shape, use_bias):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(out_shape, in_shape))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_shape))

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        elif self.use_bias:
            weight, bias = self.weights, self.bias
        else:
            weight = self.weights
            bias = None
        return F.linear(input=x, weight=weight, bias=bias)


class MetaBatchNormLayer(nn.Module):
    def __init__(self, num_features, device, args, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, meta_batch_norm=True, no_learnable_params=False,
                 use_per_step_bn_statistics=False):
        """
        A MetaBatchNorm layer. Applies the same functionality of a standard BatchNorm layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting. Also has the additional functionality of being able to store per step running stats and per step beta and gamma.
        :param num_features:
        :param device:
        :param args:
        :param eps:
        :param momentum:
        :param affine:
        :param track_running_stats:
        :param meta_batch_norm:
        :param no_learnable_params:
        :param use_per_step_bn_statistics:
        """
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.device = device
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args
        self.learnable_gamma = affine
        self.learnable_beta = affine

        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(torch.zeros(args.update_step + 1, num_features),
                                             requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(args.update_step + 1, num_features),
                                            requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(args.update_step + 1, num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(args.update_step + 1, num_features),
                                       requires_grad=self.learnable_gamma)
        else:
            self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        # if self.args.enable_inner_loop_optimizable_bn_params:
        #     self.bias = nn.Parameter(torch.zeros(num_features),
        #                              requires_grad=self.learnable_beta)
        #     self.weight = nn.Parameter(torch.ones(num_features),
        #                                requires_grad=self.learnable_gamma)

        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)

        self.momentum = momentum

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propagates by applying a bach norm function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
            # print(num_step, params['weight'])
        else:
            # print(num_step, "no params")
            weight, bias = self.weight, self.bias

        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            if (
                    params is None
                    and not self.args.enable_inner_loop_optimizable_bn_params
            ):
                bias = self.bias[num_step]
                weight = self.weight[num_step]
        else:
            running_mean = None
            running_var = None

        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)

        momentum = self.momentum

        return F.batch_norm(input, running_mean, running_var, weight, bias,
                            training=True, momentum=momentum, eps=self.eps)

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean.to(device=self.device), requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var.to(device=self.device), requires_grad=False)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class MetaLayerNormLayer(nn.Module):
    def __init__(self, input_feature_shape, eps=1e-5, elementwise_affine=True):
        """
        A MetaLayerNorm layer. A layer that applies the same functionality as a layer norm layer with the added
        capability of being able to receive params at inference time to use instead of the internal ones. As well as
        being able to use its own internal weights.
        :param input_feature_shape: The input shape without the batch dimension, e.g. c, h, w
        :param eps: Epsilon to use for protection against overflows
        :param elementwise_affine: Whether to learn a multiplicative interaction parameter 'w' in addition to
        the biases.
        """
        super(MetaLayerNormLayer, self).__init__()
        if isinstance(input_feature_shape, numbers.Integral):
            input_feature_shape = (input_feature_shape,)
        self.normalized_shape = torch.Size(input_feature_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*input_feature_shape), requires_grad=False)
            self.bias = nn.Parameter(torch.Tensor(*input_feature_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters to their initialization values.
        """
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying a layer norm function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            bias = params["bias"]
        else:
            bias = self.bias
            # print('no inner loop params', self)

        return F.layer_norm(
            input, self.normalized_shape, self.weight, bias, self.eps)

    def restore_backup_stats(self):
        pass

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class MetaConvNormLayerReLU(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False, device=None):
        """
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param args: A named tuple containing the system's hyperparameters.
           :param device: The device to run the layer on.
           :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
           :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
           meta-conv etc.
           :param input_shape: The image input shape in the form (b, c, h, w)
           :param num_filters: number of filters for convolutional layer
           :param kernel_size: the kernel size of the convolutional layer
           :param stride: the stride of the convolutional layer
           :param padding: the bias of the convolutional layer
           :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaConvNormLayerReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros(self.input_shape)

        out = x

        self.conv = MetaConv2dLayer(in_channels=out.shape[1], out_channels=self.num_filters,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding, use_bias=self.use_bias)

        out = self.conv(out)

        if self.normalization:
            if self.args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(out.shape[1], track_running_stats=True,
                                                     meta_batch_norm=self.meta_layer,
                                                     no_learnable_params=self.no_bn_learnable_params,
                                                     device=self.device,
                                                     use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                     args=self.args)
            elif self.args.norm_layer == "layer_norm":
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])

            out = self.norm_layer(out, num_step=0)

        out = F.leaky_relu(out)

        print(out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        batch_norm_params = None
        conv_params = None
        activation_function_pre_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']

                if 'activation_function_pre' in params:
                    activation_function_pre_params = params['activation_function_pre']

            conv_params = params['conv']

        out = x

        out = self.conv(out, params=conv_params)

        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          backup_running_statistics=backup_running_statistics)

        out = F.leaky_relu(out)

        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()


class MetaNormLayerConvReLU(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False, device=None):
        """
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param args: A named tuple containing the system's hyperparameters.
           :param device: The device to run the layer on.
           :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
           :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
           meta-conv etc.
           :param input_shape: The image input shape in the form (b, c, h, w)
           :param num_filters: number of filters for convolutional layer
           :param kernel_size: the kernel size of the convolutional layer
           :param stride: the stride of the convolutional layer
           :param padding: the bias of the convolutional layer
           :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaNormLayerConvReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros(self.input_shape)

        out = x
        if self.normalization:
            if self.args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(self.input_shape[1], track_running_stats=True,
                                                     meta_batch_norm=self.meta_layer,
                                                     no_learnable_params=self.no_bn_learnable_params,
                                                     device=self.device,
                                                     use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                     args=self.args)
            elif self.args.norm_layer == "layer_norm":
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])

            out = self.norm_layer.forward(out, num_step=0)
        self.conv = MetaConv2dLayer(in_channels=out.shape[1], out_channels=self.num_filters,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding, use_bias=self.use_bias)

        self.layer_dict['activation_function_pre'] = nn.LeakyReLU()

        out = self.layer_dict['activation_function_pre'].forward(self.conv.forward(out))
        print(out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        batch_norm_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization and 'norm_layer' in params:
                batch_norm_params = params['norm_layer']

            conv_params = params['conv']
        else:
            conv_params = None
            # print('no inner loop params', self)

        out = x

        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          backup_running_statistics=backup_running_statistics)

        out = self.conv.forward(out, params=conv_params)
        out = self.layer_dict['activation_function_pre'].forward(out)

        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()
