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
import net_utils


'''
Encoder architectures
'''
class ResNetEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections

    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34, 50
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(ResNetEncoder, self).__init__()

        use_bottleneck = False
        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 50:
            n_blocks = [3, 4, 6, 3]
            use_bottleneck = True
            resnet_block = net_utils.ResNetBottleneckBlock
        else:
            raise ValueError('Only supports 18, 34, 50 layer architecture')

        assert(len(n_filters) == len(n_blocks) + 1)

        activation_func = net_utils.activation_func(activation_func)

        # Resolution 1/1 -> 1/2
        in_channels, out_channels = [input_channels, n_filters[0]]
        self.conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_channels, out_channels = [n_filters[0], n_filters[1]]

        blocks2 = []
        for n in range(n_blocks[0]):
            if n == 0:
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks2.append(block)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks2.append(block)
        self.blocks2 = torch.nn.Sequential(*blocks2)

        # Resolution 1/4 -> 1/8
        blocks3 = []
        in_channels, out_channels = [n_filters[1], n_filters[2]]
        for n in range(n_blocks[1]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks3.append(block)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks3.append(block)
        self.blocks3 = torch.nn.Sequential(*blocks3)

        # Resolution 1/8 -> 1/16
        blocks4 = []
        in_channels, out_channels = [n_filters[2], n_filters[3]]
        for n in range(n_blocks[2]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks4.append(block)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks4.append(block)
        self.blocks4 = torch.nn.Sequential(*blocks4)

        # Resolution 1/16 -> 1/32
        blocks5 = []
        in_channels, out_channels = [n_filters[3], n_filters[4]]
        for n in range(n_blocks[3]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks5.append(block)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks5.append(block)
        self.blocks5 = torch.nn.Sequential(*blocks5)

    def forward(self, x):
        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks2(max_pool))

        # Resolution 1/4 -> 1/8
        layers.append(self.blocks3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.blocks4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.blocks5(layers[-1]))

        return layers[-1], layers[1:-1]


class KBNetEncoder(torch.nn.Module):
    '''
    Calibrated backprojection network (KBNet) encoder with skip connections

    Arg(s):
        in_channels_image : int
            number of input channels for image (RGB) branch
        in_channels_depth : int
            number of input channels for depth branch
        n_filters_image : int
            number of filters for image (RGB) branch for each KB layer
         n_filters_depth : int
            number of filters for depth branch  for each KB layer
        n_filters_fused : int
            number of filters for RGB 3D fusion branch  for each KB layer
        n_convolution_image : list[int]
            number of convolution layers in image branch  for each KB layer
        n_convolution_depth : list[int]
            number of convolution layers in depth branch  for each KB layer
        n_convolution_fused : list[int]
            number of convolution layers in RGB 3D fusion branch  for each KB layer
        resolutions_backprojection : list[int]
            resolutions at which to use calibrated backprojection layers
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
    '''
    def __init__(self,
                 input_channels_image=3,
                 input_channels_depth=1,
                 n_filters_image=[48, 96, 192, 384, 384],
                 n_filters_depth=[16, 32, 64, 128, 128],
                 n_filters_fused=[48, 96, 192, 384, 384],
                 n_convolutions_image=[1, 1, 1, 1, 1],
                 n_convolutions_depth=[1, 1, 1, 1, 1],
                 n_convolutions_fused=[1, 1, 1, 1, 1],
                 resolutions_backprojection=[0, 1, 2],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu'):
        super(KBNetEncoder, self).__init__()

        self.resolutions_backprojection = resolutions_backprojection

        network_depth = 5

        assert len(n_convolutions_image) == network_depth
        assert len(n_convolutions_depth) == network_depth
        assert len(n_convolutions_fused) == network_depth
        assert len(n_filters_image) == network_depth
        assert len(n_filters_depth) == network_depth
        assert len(n_filters_fused) == network_depth

        activation_func = net_utils.activation_func(activation_func)

        # Resolution: 1/1 -> 1/2
        n = 0

        if n in resolutions_backprojection:
            # Initial feature extractors on inputs
            self.conv0_image = net_utils.Conv2d(
                in_channels=input_channels_image,
                out_channels=n_filters_image[n],
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

            self.conv0_depth = net_utils.Conv2d(
                in_channels=input_channels_depth,
                out_channels=n_filters_depth[n],
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

            in_channels_image = n_filters_image[n]
            in_channels_depth = n_filters_depth[n]
            in_channels_fused = n_filters_image[n]

            self.calibrated_backprojection1 = net_utils.CalibratedBackprojectionBlock(
                in_channels_image=in_channels_image,
                in_channels_depth=in_channels_depth,
                in_channels_fused=in_channels_fused,
                n_filter_image=n_filters_image[n],
                n_filter_depth=n_filters_depth[n],
                n_filter_fused=n_filters_fused[n],
                n_convolution_image=n_convolutions_image[n],
                n_convolution_depth=n_convolutions_depth[n],
                n_convolution_fused=n_convolutions_fused[n],
                weight_initializer=weight_initializer,
                activation_func=activation_func)
        else:
            self.conv1_image = net_utils.VGGNetBlock(
                in_channels=input_channels_image,
                out_channels=n_filters_image[n],
                n_conv=n_convolutions_image[n],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

            self.conv1_depth = net_utils.VGGNetBlock(
                in_channels=input_channels_depth,
                out_channels=n_filters_depth[n],
                n_conv=n_convolutions_depth[n],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

        # Resolution: 1/2 -> 1/4
        n = 1

        in_channels_image = n_filters_image[n-1]
        in_channels_depth = n_filters_depth[n-1]

        if n in resolutions_backprojection:

            if n - 1 in resolutions_backprojection:
                in_channels_fused = n_filters_image[n-1] + n_filters_fused[n-1]
            else:
                in_channels_fused = n_filters_image[n-1]

            self.calibrated_backprojection2 = net_utils.CalibratedBackprojectionBlock(
                in_channels_image=in_channels_image,
                in_channels_depth=in_channels_depth,
                in_channels_fused=in_channels_fused,
                n_filter_image=n_filters_image[n],
                n_filter_depth=n_filters_depth[n],
                n_filter_fused=n_filters_fused[n],
                n_convolution_image=n_convolutions_image[n],
                n_convolution_depth=n_convolutions_depth[n],
                n_convolution_fused=n_convolutions_fused[n],
                weight_initializer=weight_initializer,
                activation_func=activation_func)
        else:
            self.conv2_image = net_utils.VGGNetBlock(
                in_channels=in_channels_image,
                out_channels=n_filters_image[n],
                n_conv=n_convolutions_image[n],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

            self.conv2_depth = net_utils.VGGNetBlock(
                in_channels=in_channels_depth,
                out_channels=n_filters_depth[n],
                n_conv=n_convolutions_depth[n],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

        # Resolution: 1/4 -> 1/8
        n = 2

        in_channels_image = n_filters_image[n-1]
        in_channels_depth = n_filters_depth[n-1]

        if n in resolutions_backprojection:

            if n - 1 in resolutions_backprojection:
                in_channels_fused = n_filters_image[n-1] + n_filters_fused[n-1]
            else:
                in_channels_fused = n_filters_image[n-1]

            self.calibrated_backprojection3 = net_utils.CalibratedBackprojectionBlock(
                in_channels_image=in_channels_image,
                in_channels_depth=in_channels_depth,
                in_channels_fused=in_channels_fused,
                n_filter_image=n_filters_image[n],
                n_filter_depth=n_filters_depth[n],
                n_filter_fused=n_filters_fused[n],
                n_convolution_image=n_convolutions_image[n],
                n_convolution_depth=n_convolutions_depth[n],
                n_convolution_fused=n_convolutions_fused[n],
                weight_initializer=weight_initializer,
                activation_func=activation_func)
        else:
            self.conv3_image = net_utils.VGGNetBlock(
                in_channels=in_channels_image,
                out_channels=n_filters_image[n],
                n_conv=n_convolutions_image[n],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

            self.conv3_depth = net_utils.VGGNetBlock(
                in_channels=in_channels_depth,
                out_channels=n_filters_depth[n],
                n_conv=n_convolutions_depth[n],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

        # Resolution: 1/8 -> 1/16
        n = 3

        in_channels_image = n_filters_image[n-1]
        in_channels_depth = n_filters_depth[n-1]

        if n in resolutions_backprojection:

            if n - 1 in resolutions_backprojection:
                in_channels_fused = n_filters_image[n-1] + n_filters_fused[n-1]
            else:
                in_channels_fused = n_filters_image[n-1]

            self.calibrated_backprojection4 = net_utils.CalibratedBackprojectionBlock(
                in_channels_image=in_channels_image,
                in_channels_depth=in_channels_depth,
                in_channels_fused=in_channels_fused,
                n_filter_image=n_filters_image[n],
                n_filter_depth=n_filters_depth[n],
                n_filter_fused=n_filters_fused[n],
                n_convolution_image=n_convolutions_image[n],
                n_convolution_depth=n_convolutions_depth[n],
                n_convolution_fused=n_convolutions_fused[n],
                weight_initializer=weight_initializer,
                activation_func=activation_func)
        else:
            self.conv4_image = net_utils.VGGNetBlock(
                in_channels=in_channels_image,
                out_channels=n_filters_image[n],
                n_conv=n_convolutions_image[n],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

            self.conv4_depth = net_utils.VGGNetBlock(
                in_channels=in_channels_depth,
                out_channels=n_filters_depth[n],
                n_conv=n_convolutions_depth[n],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

        # Resolution: 1/16 -> 1/32
        n = 4

        in_channels_image = n_filters_image[n-1]
        in_channels_depth = n_filters_depth[n-1]

        if n in resolutions_backprojection:

            if n - 1 in resolutions_backprojection:
                in_channels_fused = n_filters_image[n-1] + n_filters_fused[n-1]
            else:
                in_channels_fused = n_filters_image[n-1]

            self.calibrated_backprojection5 = net_utils.CalibratedBackprojectionBlock(
                in_channels_image=in_channels_image,
                in_channels_depth=in_channels_depth,
                in_channels_fused=in_channels_fused,
                n_filter_image=n_filters_image[n],
                n_filter_depth=n_filters_depth[n],
                n_filter_fused=n_filters_fused[n],
                n_convolution_image=n_convolutions_image[n],
                n_convolution_depth=n_convolutions_depth[n],
                n_convolution_fused=n_convolutions_fused[n],
                weight_initializer=weight_initializer,
                activation_func=activation_func)
        else:
            self.conv5_image = net_utils.VGGNetBlock(
                in_channels=in_channels_image,
                out_channels=n_filters_image[n],
                n_conv=n_convolutions_image[n],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

            self.conv5_depth = net_utils.VGGNetBlock(
                in_channels=in_channels_depth,
                out_channels=n_filters_depth[n],
                n_conv=n_convolutions_depth[n],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

    def forward(self, image, depth, intrinsics):

        def camera_coordinates(batch, height, width, k):
            # Reshape pixel coordinates to N x 3 x (H x W)
            xy_h = net_utils.meshgrid(
                n_batch=batch,
                n_height=height,
                n_width=width,
                device=k.device,
                homogeneous=True)
            xy_h = xy_h.view(batch, 3, -1)

            # K^-1 [x, y, 1] z and reshape back to N x 3 x H x W
            coordinates = torch.matmul(torch.inverse(k), xy_h)
            coordinates = coordinates.view(n_batch, 3, height, width)

            return coordinates

        def scale_intrinsics(batch, height0, width0, height1, width1, k):
            device = k.device

            width0 = torch.tensor(width0, dtype=torch.float32, device=device)
            height0 = torch.tensor(height0, dtype=torch.float32, device=device)
            width1 = torch.tensor(width1, dtype=torch.float32, device=device)
            height1 = torch.tensor(height1, dtype=torch.float32, device=device)

            # Get scale in x, y components
            scale_x = n_width1 / n_width0
            scale_y = n_height1 / n_height0

            # Prepare 3 x 3 matrix to do element-wise scaling
            scale = torch.tensor([[scale_x,     1.0, scale_x],
                                  [1.0,     scale_y, scale_y],
                                  [1.0,         1.0,      1.0]], dtype=torch.float32, device=device)

            scale = scale.view(1, 3, 3).repeat(n_batch, 1, 1)

            return k * scale

        layers = []

        # Resolution: 1/1 -> 1/2
        if 0 in self.resolutions_backprojection:
            n_batch, _, n_height0, n_width0 = image.shape

            # Normalized camera coordinates
            coordinates0 = camera_coordinates(n_batch, n_height0, n_width0, intrinsics)

            # Feature extractors
            conv0_image = self.conv0_image(image)
            conv0_depth = self.conv0_depth(depth)
            print(conv0_depth.shape)
            # Calibrated backprojection
            conv1_image, conv1_depth, conv1_fused = self.calibrated_backprojection1(
                image=conv0_image,
                depth=conv0_depth,
                coordinates=coordinates0,
                fused=None)

            skips1 = [conv1_fused, conv1_depth]
        else:
            conv1_image = self.conv1_image(image)
            conv1_depth = self.conv1_depth(depth)
            conv1_fused = None

            skips1 = [conv1_image, conv1_depth]

        # Store as skip connection
        layers.append(torch.cat(skips1, dim=1))

        # Resolution: 1/2 -> 1/4
        _, _, n_height1, n_width1 = conv1_image.shape

        if 1 in self.resolutions_backprojection:
            intrinsics1 = scale_intrinsics(
                batch=n_batch,
                height0=n_height0,
                width0=n_width0,
                height1=n_height1,
                width1=n_width1,
                k=intrinsics)

            # Normalized camera coordinates
            coordinates1 = camera_coordinates(n_batch, n_height1, n_width1, intrinsics1)

            # Calibrated backprojection
            conv2_image, conv2_depth, conv2_fused = self.calibrated_backprojection2(
                image=conv1_image,
                depth=conv1_depth,
                coordinates=coordinates1,
                fused=conv1_fused)

            skips2 = [conv2_fused, conv2_depth]
        else:
            if conv1_fused is not None:
                conv2_image = self.conv2_image(conv1_fused)
            else:
                conv2_image = self.conv2_image(conv1_image)

            conv2_depth = self.conv2_depth(conv1_depth)
            conv2_fused = None

            skips2 = [conv2_image, conv2_depth]

        # Store as skip connection
        layers.append(torch.cat(skips2, dim=1))

        # Resolution: 1/4 -> 1/8
        _, _, n_height2, n_width2 = conv2_image.shape

        if 2 in self.resolutions_backprojection:
            intrinsics2 = scale_intrinsics(
                batch=n_batch,
                height0=n_height0,
                width0=n_width0,
                height1=n_height2,
                width1=n_width2,
                k=intrinsics)

            # Normalized camera coordinates
            coordinates2 = camera_coordinates(n_batch, n_height2, n_width2, intrinsics2)

            # Calibrated backprojection
            conv3_image, conv3_depth, conv3_fused = self.calibrated_backprojection3(
                image=conv2_image,
                depth=conv2_depth,
                coordinates=coordinates2,
                fused=conv2_fused)

            skips3 = [conv3_fused, conv3_depth]
        else:
            if conv2_fused is not None:
                conv3_image = self.conv3_image(conv2_fused)
            else:
                conv3_image = self.conv3_image(conv2_image)

            conv3_depth = self.conv3_depth(conv2_depth)
            conv3_fused = None

            skips3 = [conv3_image, conv3_depth]

        # Store as skip connection
        layers.append(torch.cat(skips3, dim=1))

        # Resolution: 1/8 -> 1/16
        _, _, n_height3, n_width3 = conv3_image.shape

        if 3 in self.resolutions_backprojection:
            intrinsics3 = scale_intrinsics(
                batch=n_batch,
                height0=n_height0,
                width0=n_width0,
                height1=n_height3,
                width1=n_width3,
                k=intrinsics)

            # Normalized camera coordinates
            coordinates3 = camera_coordinates(n_batch, n_height3, n_width3, intrinsics3)

            # Calibrated backprojection
            conv4_image, conv4_depth, conv4_fused = self.calibrated_backprojection4(
                image=conv3_image,
                depth=conv3_depth,
                coordinates=coordinates3,
                fused=conv3_fused)

            skips4 = [conv4_fused, conv4_depth]
        else:
            if conv3_fused is not None:
                conv4_image = self.conv4_image(conv3_fused)
            else:
                conv4_image = self.conv4_image(conv3_image)

            conv4_depth = self.conv4_depth(conv3_depth)
            conv4_fused = None

            skips4 = [conv4_image, conv4_depth]

        # Store as skip connection
        layers.append(torch.cat(skips4, dim=1))

        # Resolution: 1/16 -> 1/32
        _, _, n_height4, n_width4 = conv4_image.shape

        if 4 in self.resolutions_backprojection:
            intrinsics4 = scale_intrinsics(
                batch=n_batch,
                height0=n_height0,
                width0=n_width0,
                height1=n_height4,
                width1=n_width4,
                k=intrinsics)

            # Normalized camera coordinates
            coordinates4 = camera_coordinates(n_batch, n_height4, n_width4, intrinsics4)

            # Calibrated backprojection
            conv5_image, conv5_depth, conv5_fused = self.calibrated_backprojection4(
                image=conv4_image,
                depth=conv4_depth,
                coordinates=coordinates4,
                fused=conv4_fused)

            skips5 = [conv5_fused, conv5_depth]
        else:
            if conv4_fused is not None:
                conv5_image = self.conv5_image(conv4_fused)
            else:
                conv5_image = self.conv5_image(conv4_image)

            conv5_depth = self.conv5_depth(conv4_depth)
            conv5_fused = None

            skips5 = [conv5_image, conv5_depth]

        # Store as skip connection
        layers.append(torch.cat(skips5, dim=1))

        return layers[-1], layers[0:-1]


class PoseEncoder(torch.nn.Module):
    '''
    Pose network encoder

    Arg(s):
        input_channels : int
            number of channels in input data
        n_filters : list[int]
            number of filters to use for each convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 input_channels=6,
                 n_filters=[16, 32, 64, 128, 256, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(PoseEncoder, self).__init__()

        activation_func = net_utils.activation_func(activation_func)

        self.conv1 = net_utils.Conv2d(
            input_channels,
            n_filters[0],
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv2 = net_utils.Conv2d(
            n_filters[0],
            n_filters[1],
            kernel_size=5,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv3 = net_utils.Conv2d(
            n_filters[1],
            n_filters[2],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv4 = net_utils.Conv2d(
            n_filters[2],
            n_filters[3],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv5 = net_utils.Conv2d(
            n_filters[3],
            n_filters[4],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv6 = net_utils.Conv2d(
            n_filters[4],
            n_filters[5],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv7 = net_utils.Conv2d(
            n_filters[5],
            n_filters[6],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

    def forward(self, x):
        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        layers.append(self.conv2(layers[-1]))

        # Resolution 1/4 -> 1/8
        layers.append(self.conv3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.conv4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.conv5(layers[-1]))

        # Resolution 1/32 -> 1/64
        layers.append(self.conv6(layers[-1]))

        # Resolution 1/64 -> 1/128
        layers.append(self.conv7(layers[-1]))

        return layers[-1], None


'''
Decoder architectures
'''
class MultiScaleDecoder(torch.nn.Module):
    '''
    Multi-scale decoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels or classes in output
        n_scale : int
            number of output scales for multi-scale prediction
        n_filters : list[int]
            number of filters to use at each decoder block
        n_skips : list[int[]]
            number of filters from skip connections
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        output_func : func
            activation function for output
        output_kernel_sizes : list[int]
            list of kernel size in output module
        use_batch_norm : bool
            if set, then applied batch normalization
        deconv_type : str
            deconvolution types available: transpose, up
    '''

    def __init__(self,
                 input_channels=256,
                 output_channels=1,
                 n_scale=4,
                 n_filters=[256, 128, 64, 32, 16],
                 n_skips=[256, 128, 64, 32, 0],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 output_func='linear',
                 use_batch_norm=False,
                 deconv_type='transpose'):
        super(MultiScaleDecoder, self).__init__()

        network_depth = 5
        assert(n_scale > 0 and n_scale < network_depth)
        assert(len(n_filters) == network_depth)
        assert(len(n_skips) == network_depth)

        self.n_scale = n_scale
        self.output_func = output_func

        activation_func = net_utils.activation_func(activation_func)
        output_func = net_utils.activation_func(output_func)

        # Upsampling from lower to full resolution requires multi-scale
        if 'upsample' in self.output_func and self.n_scale < 2:
            self.n_scale = 2

        # Resolution 1/32 -> 1/16
        in_channels, skip_channels, out_channels = [
            input_channels, n_skips[0], n_filters[0]
        ]
        self.deconv4 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        # Resolution 1/16 -> 1/8
        in_channels, skip_channels, out_channels = [
            n_filters[0], n_skips[1], n_filters[1]
        ]
        self.deconv3 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        if self.n_scale > 3:
            self.output3 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

        # Resolution 1/8 -> 1/4
        in_channels, skip_channels, out_channels = [
            n_filters[1], n_skips[2], n_filters[2]
        ]
        if self.n_scale > 3:
            skip_channels = skip_channels + output_channels

        self.deconv2 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        if self.n_scale > 2:
            self.output2 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

        # Resolution 1/4 -> 1/2
        in_channels, skip_channels, out_channels = [
            n_filters[2], n_skips[3], n_filters[3]
        ]
        if self.n_scale > 2:
            skip_channels = skip_channels + output_channels

        self.deconv1 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        if self.n_scale > 1:
            self.output1 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

        # Resolution 1/2 -> 1/1
        if 'upsample' not in self.output_func:
            in_channels, skip_channels, out_channels = [
                n_filters[3], n_skips[4], n_filters[4]
            ]
            if self.n_scale > 1:
                skip_channels = skip_channels + output_channels

            self.deconv0 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                deconv_type=deconv_type)

            self.output0 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

    def forward(self, x, skips):
        layers = [x]
        outputs = []

        # Resolution 1/32 -> 1/16
        n = len(skips) - 1
        layers.append(self.deconv4(layers[-1], skips[n]))

        # Resolution 1/16 -> 1/8
        n = n - 1
        layers.append(self.deconv3(layers[-1], skips[n]))

        if self.n_scale > 3:
            output3 = self.output3(layers[-1])
            outputs.append(output3)
            upsample_output3 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='nearest')

        # Resolution 1/8 -> 1/4
        n = n - 1
        skip = torch.cat([skips[n], upsample_output3], dim=1) if self.n_scale > 3 else skips[n]
        layers.append(self.deconv2(layers[-1], skip))

        if self.n_scale > 2:
            output2 = self.output2(layers[-1])
            outputs.append(output2)
            upsample_output2 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='nearest')

        # Resolution 1/4 -> 1/2
        n = n - 1
        skip = torch.cat([skips[n], upsample_output2], dim=1) if self.n_scale > 2 else skips[n]
        layers.append(self.deconv1(layers[-1], skip))

        if self.n_scale > 1:
            output1 = self.output1(layers[-1])
            outputs.append(output1)
            upsample_output1 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='nearest')

        # Resolution 1/2 -> 1/1
        n = n - 1

        if 'upsample' in self.output_func:
            output0 = upsample_output1
        else:
            if self.n_scale > 1:
                # If there is skip connection at layer 0
                skip = torch.cat([skips[n], upsample_output1], dim=1) if n == 0 else upsample_output1
                layers.append(self.deconv0(layers[-1], skip))
            else:
                if n == 0:
                    layers.append(self.deconv0(layers[-1], skips[n]))
                else:
                    layers.append(self.deconv0(layers[-1]))

            output0 = self.output0(layers[-1])

        outputs.append(output0)

        return outputs


class PoseDecoder(torch.nn.Module):
    '''
    Pose Decoder 6 DOF

    Arg(s):
        rotation_parameterization : str
            axis
        input_channels : int
            number of channels in input latent vector
        n_filters : int list
            number of filters to use at each decoder block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 rotation_parameterization,
                 input_channels=256,
                 n_filters=[],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(PoseDecoder, self).__init__()

        self.rotation_parameterization = rotation_parameterization

        activation_func = net_utils.activation_func(activation_func)

        if len(n_filters) > 0:
            layers = []
            in_channels = input_channels

            for out_channels in n_filters:
                conv = net_utils.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                layers.append(conv)
                in_channels = out_channels

            conv = net_utils.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=None,
                use_batch_norm=False)
            layers.append(conv)

            self.conv = torch.nn.Sequential(*layers)
        else:
            self.conv = net_utils.Conv2d(
                in_channels=input_channels,
                out_channels=6,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=None,
                use_batch_norm=False)

    def forward(self, x):
        conv_output = self.conv(x)
        pose_mean = torch.mean(conv_output, [2, 3])
        dof = 0.01 * pose_mean
        posemat = net_utils.pose_matrix(
            dof,
            rotation_parameterization=self.rotation_parameterization)

        return posemat


class SparseToDensePool(torch.nn.Module):
    '''
    Converts sparse inputs to dense outputs using max and min pooling
    with different kernel sizes and combines them with 1 x 1 convolutions

    Arg(s):
        input_channels : int
            number of channels to be fed to max and/or average pool(s)
        min_pool_sizes : list[int]
            list of min pool sizes s (kernel size is s x s)
        max_pool_sizes : list[int]
            list of max pool sizes s (kernel size is s x s)
        n_filter : int
            number of filters for 1 x 1 convolutions
        n_convolution : int
            number of 1 x 1 convolutions to use for balancing detail and density
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
    '''

    def __init__(self,
                 input_channels,
                 min_pool_sizes=[3, 5, 7, 9],
                 max_pool_sizes=[3, 5, 7, 9],
                 n_filter=8,
                 n_convolution=3,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu'):
        super(SparseToDensePool, self).__init__()

        activation_func = net_utils.activation_func(activation_func)

        self.min_pool_sizes = [
            s for s in min_pool_sizes if s > 1
        ]

        self.max_pool_sizes = [
            s for s in max_pool_sizes if s > 1
        ]

        # Construct min pools
        self.min_pools = []
        for s in self.min_pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.min_pools.append(pool)

        # Construct max pools
        self.max_pools = []
        for s in self.max_pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.max_pools.append(pool)

        self.len_pool_sizes = len(self.min_pool_sizes) + len(self.max_pool_sizes)

        in_channels = len(self.min_pool_sizes) + len(self.max_pool_sizes)

        pool_convs = []
        for n in range(n_convolution):
            conv = net_utils.Conv2d(
                in_channels,
                n_filter,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=False)
            pool_convs.append(conv)

            # Set new input channels as output channels
            in_channels = n_filter

        self.pool_convs = torch.nn.Sequential(*pool_convs)

        in_channels = n_filter + input_channels

        self.conv = net_utils.Conv2d(
            in_channels,
            n_filter,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=False)

    def forward(self, x):
        # Input depth
        z = torch.unsqueeze(x[:, 0, ...], dim=1)

        pool_pyramid = []

        # Use min and max pooling to densify and increase receptive field
        for pool, s in zip(self.min_pools, self.min_pool_sizes):
            # Set flag (999) for any zeros and max pool on -z then revert the values
            z_pool = -pool(torch.where(z == 0, -999 * torch.ones_like(z), -z))
            # Remove any 999 from the results
            z_pool = torch.where(z_pool == 999, torch.zeros_like(z), z_pool)

            pool_pyramid.append(z_pool)

        for pool, s in zip(self.max_pools, self.max_pool_sizes):
            z_pool = pool(z)

            pool_pyramid.append(z_pool)

        # Stack max and minpools into pyramid
        pool_pyramid = torch.cat(pool_pyramid, dim=1)

        # Learn weights for different kernel sizes, and near and far structures
        pool_convs = self.pool_convs(pool_pyramid)

        pool_convs = torch.cat([pool_convs, x], dim=1)

        return self.conv(pool_convs)
