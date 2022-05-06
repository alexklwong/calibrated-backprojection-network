'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
import torch


EPSILON = 1e-10


def activation_func(activation_fn):
    '''
    Select activation function

    Arg(s):
        activation_fn : str
            name of activation function
    Returns:
        torch.nn.Module : activation function
    '''

    if 'linear' in activation_fn:
        return None
    elif 'leaky_relu' in activation_fn:
        return torch.nn.LeakyReLU(negative_slope=0.20, inplace=True)
    elif 'relu' in activation_fn:
        return torch.nn.ReLU()
    elif 'elu' in activation_fn:
        return torch.nn.ELU()
    elif 'sigmoid' in activation_fn:
        return torch.nn.Sigmoid()
    else:
        raise ValueError('Unsupported activation function: {}'.format(activation_fn))


'''
Network layers
'''
class Conv2d(torch.nn.Module):
    '''
    2D convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(Conv2d, self).__init__()

        padding = kernel_size // 2

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through a convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        conv = self.conv(x)

        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class DepthwiseSeparableConv2d(torch.nn.Module):
    '''
    Depthwise separable convolution class
    Performs
    1. separate k x k convolution per channel (depth-wise)
    2. 1 x 1 convolution across all channels (point-wise)

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel (k x k)
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(DepthwiseSeparableConv2d, self).__init__()

        padding = kernel_size // 2

        self.conv_depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=in_channels)

        self.conv_pointwise = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv_depthwise.weight)
            torch.nn.init.kaiming_normal_(self.conv_pointwise.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv_depthwise.weight)
            torch.nn.init.xavier_normal_(self.conv_pointwise.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv_depthwise.weight)
            torch.nn.init.xavier_uniform_(self.conv_pointwise.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.conv = torch.nn.Sequential(
            self.conv_depthwise,
            self.conv_pointwise)

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

        self.activation_func = activation_func

    def forward(self, x):
        '''
        Forward input x through a depthwise convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        conv = self.conv(x)

        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class AtrousConv2d(torch.nn.Module):
    '''
    2D atrous convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        dilation : int
            dilation of convolution (skips rate - 1 pixels)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(AtrousConv2d, self).__init__()

        padding = dilation

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding=padding,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through an atrous convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        conv = self.conv(x)

        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class TransposeConv2d(torch.nn.Module):
    '''
    Transpose convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(TransposeConv2d, self).__init__()

        padding = kernel_size // 2

        self.deconv = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.deconv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.deconv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.deconv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through a transposed convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        deconv = self.deconv(x)

        if self.use_batch_norm:
            deconv = self.batch_norm(deconv)
        elif self.use_instance_norm:
            deconv = self.instance_norm(deconv)

        if self.activation_func is not None:
            return self.activation_func(deconv)
        else:
            return deconv


class UpConv2d(torch.nn.Module):
    '''
    Up-convolution (upsample + convolution) block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        shape : list[int]
            two element tuple of ints (height, width)
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(UpConv2d, self).__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x, shape):
        '''
        Forward input x through an up convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        upsample = torch.nn.functional.interpolate(x, size=shape, mode='nearest')
        conv = self.conv(upsample)
        return conv


class FullyConnected(torch.nn.Module):
    '''
    Fully connected layer

    Arg(s):
        in_channels : int
            number of input neurons
        out_channels : int
            number of output neurons
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        dropout_rate : float
            probability to use dropout
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 dropout_rate=0.00):
        super(FullyConnected, self).__init__()

        self.fully_connected = torch.nn.Linear(in_features, out_features)

        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.fully_connected.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.fully_connected.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.fully_connected.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        if dropout_rate > 0.00 and dropout_rate <= 1.00:
            self.dropout = torch.nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        '''
        Forward input x through a fully connected block

        Arg(s):
            x : torch.Tensor[float32]
                N x C input tensor
        Returns:
            torch.Tensor[float32] : N x K output tensor
        '''

        fully_connected = self.fully_connected(x)

        if self.activation_func is not None:
            fully_connected = self.activation_func(fully_connected)

        if self.dropout is not None:
            return self.dropout(fully_connected)
        else:
            return fully_connected


'''
Network encoder blocks
'''
class ResNetBlock(torch.nn.Module):
    '''
    Basic ResNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(ResNetBlock, self).__init__()

        self.activation_func = activation_func

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.projection = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through a basic ResNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv2 + X)


class ResNetBottleneckBlock(torch.nn.Module):
    '''
    ResNet bottleneck block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(ResNetBottleneckBlock, self).__init__()

        self.activation_func = activation_func

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv3 = conv2d(
            out_channels,
            4 * out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.projection = Conv2d(
            in_channels,
            4 * out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through a ResNet bottleneck block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv3 + X)


class AtrousResNetBlock(torch.nn.Module):
    '''
    Basic atrous ResNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dilation : int
            dilation of convolution (skips rate - 1 pixels)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation=2,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(AtrousResNetBlock, self).__init__()

        self.activation_func = activation_func

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        self.conv1 = AtrousConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.projection = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through an atrous ResNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)

        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv2 + X)


class VGGNetBlock(torch.nn.Module):
    '''
    VGGNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_convolution : int
            number of convolution layers
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_convolution=1,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(VGGNetBlock, self).__init__()

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        layers = []
        for n in range(n_convolution - 1):
            conv = conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            layers.append(conv)
            in_channels = out_channels

        conv = conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)
        layers.append(conv)

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward input x through a VGG block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        return self.conv_block(x)


class AtrousVGGNetBlock(torch.nn.Module):
    '''
    Atrous VGGNet block class
    (last block performs atrous convolution instead of convolution with stride)

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_convolution : int
            number of convolution layers
        dilation : int
            dilation of atrous convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_convolution=1,
                 dilation=2,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(AtrousVGGNetBlock, self).__init__()

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        layers = []
        for n in range(n_convolution - 1):
            conv = conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            layers.append(conv)
            in_channels = out_channels

        conv = AtrousConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)
        layers.append(conv)

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward input x through an atrous VGG block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        return self.conv_block(x)


class AtrousSpatialPyramidPooling(torch.nn.Module):
    '''
    Atrous Spatial Pyramid Pooling class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dilations : list[int]
            dilations for different atrous convolution of each branch
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations=[6, 12, 18],
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(AtrousSpatialPyramidPooling, self).__init__()

        output_channels = out_channels // (len(dilations) + 1)

        # Point-wise 1 by 1 convolution branch
        self.conv1 = Conv2d(
            in_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # 3 by 3 convolutions with different dilation rate
        self.atrous_convs = torch.nn.ModuleList()

        for dilation in dilations:
            atrous_conv = AtrousConv2d(
                in_channels,
                output_channels,
                kernel_size=3,
                dilation=dilation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            self.atrous_convs.append(atrous_conv)

        # Global pooling
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.global_pool_conv = Conv2d(
            in_channels,
            output_channels,
            kernel_size=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Fuse point-wise (1 by 1) convolution, atrous convolutions, global pooling
        self.conv_fuse = Conv2d(
            (len(dilations) + 2) * output_channels,
            out_channels,
            kernel_size=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through a ASPP block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # List to hold all branches
        branches = []

        # Point-wise (1 by 1) convolution branch
        branches.append(self.conv1(x))

        # Atrous branches
        for atrous_conv in self.atrous_convs:
            branches.append(atrous_conv(x))

        # Global pooling branch
        global_pool = self.global_pool(x)
        global_pool = self.global_pool_conv(global_pool)
        global_pool = torch.nn.functional.interpolate(
            global_pool,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=True)
        branches.append(global_pool)

        return self.conv_fuse(torch.cat(branches, dim=1))


class SpatialPyramidPooling(torch.nn.Module):
    '''
    Spatial Pyramid Pooling class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_sizes : list[int]
            pooling kernel size of each branch
        pool_func : str
            max, average
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 pool_func='max',
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(SpatialPyramidPooling, self).__init__()

        output_channels = out_channels // len(kernel_sizes)

        # List of pooling kernel sizes
        self.kernel_sizes = kernel_sizes

        if pool_func == 'max':
            self.pool_func = torch.nn.functional.max_pool2d
        elif pool_func == 'average':
            self.pool_func = torch.nn.functional.avg_pool2d
        else:
            raise ValueError('Unsupported pooling function: {}'.format(pool_func))

        # List of convolutions to compress feature maps
        self.convs = torch.nn.ModuleList()

        for n in kernel_sizes:
            conv = Conv2d(
                in_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            self.convs.append(conv)

        self.conv_fuse = torch.nn.Sequential(
            Conv2d(
                2 * len(kernel_sizes) * output_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                weight_initializer=weight_initializer,
                activation_func=None,
                use_batch_norm=False,
                use_instance_norm=False))

    def forward(self, x):
        '''
        Forward input x through SPP block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # List to hold all branches
        branches = [x]

        # Pyramid pooling branches
        for kernel_size, conv in zip(self.kernel_sizes, self.convs):
            pool = self.pool_func(
                x,
                kernel_size=(kernel_size, kernel_size),
                stride=(kernel_size, kernel_size))

            pool = torch.nn.functional.interpolate(
                pool,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=True)

            branches.append(conv(pool))

        return self.conv_fuse(torch.cat(branches, dim=1))


class CalibratedBackprojectionBlock(torch.nn.Module):
    '''
    Calibrated backprojection (KB) layer class

    Arg(s):
        in_channels_image : int
            number of input channels for image (RGB) branch
        in_channels_depth : int
            number of input channels for depth branch
        in_channels_fused : int
            number of input channels for RGB 3D fusion branch
        n_filter_image : int
            number of filters for image (RGB) branch
        n_filter_depth : int
            number of filters for depth branch
        n_filter_fused : int
            number of filters for RGB 3D fusion branch
        n_convolution_image : int
            number of convolution layers in image branch
        n_convolution_depth : int
            number of convolution layers in depth branch
        n_convolution_fused : int
            number of convolution layers in RGB 3D fusion branch
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
    '''
    def __init__(self,
                 in_channels_image,
                 in_channels_depth,
                 in_channels_fused,
                 n_filter_image=48,
                 n_filter_depth=16,
                 n_filter_fused=48,
                 n_convolution_image=1,
                 n_convolution_depth=1,
                 n_convolution_fused=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)):
        super(CalibratedBackprojectionBlock, self).__init__()

        self.conv_image = VGGNetBlock(
            in_channels=in_channels_image,
            out_channels=n_filter_image,
            n_convolution=n_convolution_image,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.conv_depth = VGGNetBlock(
            in_channels=in_channels_depth + 3,
            out_channels=n_filter_depth,
            n_convolution=n_convolution_depth,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.proj_depth = Conv2d(
            in_channels=in_channels_depth,
            out_channels=1,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.conv_fused = Conv2d(
            in_channels=in_channels_fused + 3,
            out_channels=n_filter_fused,
            kernel_size=1,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

    def forward(self, image, depth, coordinates, fused=None):

        layers_fused = []

        # Obtain image (RGB) features
        conv_image = self.conv_image(image)

        # Obtain depth (Z) features
        conv_depth = self.conv_depth(torch.cat([depth, coordinates], dim=1))

        # Include image (RGB) features
        layers_fused.append(image)

        # Project depth features to 1 dimension
        z = self.proj_depth(depth)

        # Include backprojected 3D positional (XYZ) encoding: K^-1 [x y 1] z
        xyz = coordinates * z
        layers_fused.append(xyz)

        # Include previous RGBXYZ representation
        if fused is not None:
            layers_fused.append(fused)

        # Obtain fused (RGBXYZ) representation
        layers_fused = torch.cat(layers_fused, dim=1)
        conv_fused = self.conv_fused(layers_fused)

        return conv_image, conv_depth, conv_fused


'''
Network decoder blocks
'''
class DecoderBlock(torch.nn.Module):
    '''
    Decoder block with skip connection

    Arg(s):
        in_channels : int
            number of input channels
        skip_channels : int
            number of skip connection channels
        out_channels : int
            number of output channels
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        deconv_type : str
            deconvolution types: transpose, up
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 deconv_type='up',
                 use_depthwise_separable=False):
        super(DecoderBlock, self).__init__()

        self.skip_channels = skip_channels
        self.deconv_type = deconv_type

        if deconv_type == 'transpose':
            self.deconv = TransposeConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
        elif deconv_type == 'up':
            self.deconv = UpConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)

        concat_channels = skip_channels + out_channels

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        self.conv = conv2d(
            concat_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x, skip=None, shape=None):
        '''
        Forward input x through a decoder block and fuse with skip connection

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            skip : torch.Tensor[float32]
                N x F x H x W skip connection
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        if self.deconv_type == 'transpose':
            deconv = self.deconv(x)
        elif self.deconv_type == 'up':

            if skip is not None:
                shape = skip.shape[2:4]
            elif shape is not None:
                pass
            else:
                n_height, n_width = x.shape[2:4]
                shape = (int(2 * n_height), int(2 * n_width))

            deconv = self.deconv(x, shape=shape)

        if self.skip_channels > 0:
            concat = torch.cat([deconv, skip], dim=1)
        else:
            concat = deconv

        return self.conv(concat)


'''
Pose regression layer
'''
def pose_matrix(v, rotation_parameterization='axis'):
    '''
    Convert 6 DoF parameters to transformation matrix

    Arg(s):
        v : torch.Tensor[float32]
            N x 6 vector in the order of tx, ty, tz, rx, ry, rz
        rotation_parameterization : str
            axis
    Returns:
        torch.Tensor[float32] : N x 4 x 4 homogeneous transformation matrix
    '''

    # Select N x 3 element rotation vector
    r = v[..., :3]
    # Select N x 3 element translation vector
    t = v[..., 3:]

    if rotation_parameterization == 'axis':
        Rt = transformation_from_parameters(torch.unsqueeze(r, dim=1), t)
    else:
        raise ValueError('Unsupported rotation parameterization: {}'.format(rotation_parameterization))

    return Rt


'''
Utility functions for rotation
'''
def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


'''
Utility functions for rigid warping
'''
def meshgrid(n_batch, n_height, n_width, device, homogeneous=True):
    '''
    Creates N x 2 x H x W meshgrid in x, y directions

    Arg(s):
        n_batch : int
            batch size
        n_height : int
            height of tensor
        n_width : int
            width of tensor
        device : torch.device
            device on which to create meshgrid
        homoegenous : bool
            if set, then add homogeneous coordinates (N x H x W x 3)
    Return:
        torch.Tensor[float32]: N x 2 x H x W meshgrid of x, y and 1 (if homogeneous)
    '''

    x = torch.linspace(start=0.0, end=n_width-1, steps=n_width, device=device)
    y = torch.linspace(start=0.0, end=n_height-1, steps=n_height, device=device)

    # Create H x W grids
    grid_y, grid_x = torch.meshgrid(y, x)

    if homogeneous:
        # Create 3 x H x W grid (x, y, 1)
        grid_xy = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
    else:
        # Create 2 x H x W grid (x, y)
        grid_xy = torch.stack([grid_x, grid_y], dim=0)

    grid_xy = torch.unsqueeze(grid_xy, dim=0) \
        .repeat(n_batch, 1, 1, 1)

    return grid_xy

def backproject_to_camera(depth, intrinsics, shape):
    '''
    Backprojects pixel coordinates to 3D camera coordinates

    Arg(s):
        depth : torch.Tensor[float32]
            N x 1 x H x W depth map
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 4 x (H x W)
    '''
    n_batch, _, n_height, n_width = shape

    # Create homogeneous coordinates [x, y, 1]
    xy_h = meshgrid(n_batch, n_height, n_width, device=depth.device, homogeneous=True)

    # Reshape pixel coordinates to N x 3 x (H x W)
    xy_h = xy_h.view(n_batch, 3, -1)

    # Reshape depth as N x 1 x (H x W)
    depth = depth.view(n_batch, 1, -1)

    # K^-1 [x, y, 1] z
    points = torch.matmul(torch.inverse(intrinsics), xy_h) * depth

    # Make homogeneous
    return torch.cat([points, torch.ones_like(depth)], dim=1)

def project_to_pixel(points, pose, intrinsics, shape):
    '''
    Projects points in camera coordinates to 2D pixel coordinates

    Arg(s):
        points : torch.Tensor[float32]
            N x 4 x (H x W) depth map
        pose : torch.Tensor[float32]
            N x 4 x 4 transformation matrix
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 2 x H x W
    '''

    n_batch, _, n_height, n_width = shape

    # Convert camera intrinsics to homogeneous coordinates
    column = torch.zeros([n_batch, 3, 1], device=points.device)
    row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=points.device) \
        .view(1, 1, 4) \
        .repeat(n_batch, 1, 1)
    intrinsics = torch.cat([intrinsics, column], dim=2)
    intrinsics = torch.cat([intrinsics, row], dim=1)

    # Apply the transformation and project: \pi K g p
    T = torch.matmul(intrinsics, pose)
    T = T[:, 0:3, :]
    points = torch.matmul(T, points)
    points = points / (torch.unsqueeze(points[:, 2, :], dim=1) + 1e-7)
    points = points[:, 0:2, :]

    # Reshape to N x 2 x H x W
    return points.view(n_batch, 2, n_height, n_width)

def grid_sample(image, target_xy, shape, padding_mode='border'):
    '''
    Samples the image at x, y locations to target x, y locations

    Arg(s):
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
        target_xy : torch.Tensor[float32]
            N x 2 x H x W target x, y locations in image space
        shape : list[int]
            shape of tensor in (N, C, H, W)
        padding_mode : str
            padding to use when sampled out of bounds
    Return:
        torch.Tensor[float32] : N x 3 x H x W RGB image
    '''

    n_batch, _, n_height, n_width = shape

    # Swap dimensions to N x H x W x 2 for grid sample
    target_xy = target_xy.permute(0, 2, 3, 1)

    # Normalize coordinates between -1 and 1
    target_xy[..., 0] /= (n_width - 1.0)
    target_xy[..., 1] /= (n_height - 1.0)
    target_xy = 2.0 * (target_xy - 0.5)

    # Sample the image at normalized target x, y locations
    return torch.nn.functional.grid_sample(
        image,
        grid=target_xy,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True)


'''
Utility function to pre-process sparse depth
'''
class OutlierRemoval(object):
    '''
    Class to perform outlier removal based on depth difference in local neighborhood

    Arg(s):
        kernel_size : int
            local neighborhood to consider
        threshold : float
            depth difference threshold
    '''

    def __init__(self, kernel_size=7, threshold=1.5):

        self.kernel_size = kernel_size
        self.threshold = threshold

    def remove_outliers(self, sparse_depth, validity_map):
        '''
        Removes erroneous measurements from sparse depth and validity map

        Arg(s):
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W tensor sparse depth
            validity_map : torch.Tensor[float32]
                N x 1 x H x W tensor validity map
        Returns:
            torch.Tensor[float32] : N x 1 x H x W sparse depth
            torch.Tensor[float32] : N x 1 x H x W validity map
        '''

        # Replace all zeros with large values
        max_value = 10 * torch.max(sparse_depth)
        sparse_depth_max_filled = torch.where(
            validity_map <= 0,
            torch.full_like(sparse_depth, fill_value=max_value),
            sparse_depth)

        # For each neighborhood find the smallest value
        padding = self.kernel_size // 2
        sparse_depth_max_filled = torch.nn.functional.pad(
            input=sparse_depth_max_filled,
            pad=(padding, padding, padding, padding),
            mode='constant',
            value=max_value)

        min_values = -torch.nn.functional.max_pool2d(
            input=-sparse_depth_max_filled,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0)

        # If measurement differs a lot from minimum value then remove
        validity_map_clean = torch.where(
            min_values < sparse_depth - self.threshold,
            torch.zeros_like(validity_map),
            torch.ones_like(validity_map))

        # Update sparse depth and validity map
        validity_map_clean = validity_map * validity_map_clean
        sparse_depth_clean = sparse_depth * validity_map_clean

        return sparse_depth_clean, validity_map_clean
