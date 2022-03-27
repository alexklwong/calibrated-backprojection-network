import argparse
import global_constants as settings
# from kbnet import run
from PIL import Image
import datasets, data_utils, eval_utils
from log_utils import log
from kbnet_model import KBNetModel
from posenet_model import PoseNetModel
import global_constants as settings
from transforms import Transforms
from net_utils import OutlierRemoval


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',
        type=str, required=True, help='Path to list of image paths')
    parser.add_argument('--sparse_depth_path',
        type=str, required=True, help='Path to list of sparse depth paths')
    parser.add_argument('--intrinsics_path',
        type=str, required=True, help='Path to list of camera intrinsics paths')
    parser.add_argument('--input_channels_image',
        type=int, default=3, help='Number of input image channels')
    parser.add_argument('--input_channels_depth',
        type=int, default=2, help='Number of input depth channels')
    parser.add_argument('--normalized_image_range',
        nargs='+', type=float, default=[0, 1], help='Range of image intensities after normalization')
    parser.add_argument('--outlier_removal_kernel_size',
        type=int, default=7, help='Kernel size to filter outlier sparse depth')
    parser.add_argument('--outlier_removal_threshold',
        type=float, default=1.5, help='Difference threshold to consider a point an outlier')
    # Sparse to dense pool settings
    parser.add_argument('--min_pool_sizes_sparse_to_dense_pool',
        nargs='+', type=int, default=[3, 7, 9, 11], help='Space delimited list of min pool sizes for sparse to dense pooling')
    parser.add_argument('--max_pool_sizes_sparse_to_dense_pool',
        nargs='+', type=int, default=[3, 7, 9, 11], help='Space delimited list of max pool sizes for sparse to dense pooling')
    parser.add_argument('--n_convolution_sparse_to_dense_pool',
        type=int, default=3, help='Number of convolutions for sparse to dense pooling')
    parser.add_argument('--n_filter_sparse_to_dense_pool',
        type=int, default=8, help='Number of filters for sparse to dense pooling')
    # Depth network settings
    parser.add_argument('--n_filters_encoder_image',
        nargs='+', type=int, default=[48, 96, 192, 384, 384], help='Space delimited list of filters to use in each block of image encoder')
    parser.add_argument('--n_filters_encoder_depth',
        nargs='+', type=int, default=[16, 32, 64, 128, 128], help='Space delimited list of filters to use in each block of depth encoder')
    parser.add_argument('--resolutions_backprojection',
        nargs='+', type=int, default=[0, 1, 2, 3], help='Space delimited list of resolutions to use calibrated backprojection')
    parser.add_argument('--n_filters_decoder',
        nargs='+', type=int, default=[256, 128, 128, 64, 12], help='Space delimited list of filters to use in each block of depth decoder')
    parser.add_argument('--deconv_type',
        type=str, default='up', help='Deconvolution type: up, transpose')
    parser.add_argument('--min_predict_depth',
        type=float, default=0.5, help='Minimum value of predicted depth')
    parser.add_argument('--max_predict_depth',
        type=float, default=100.0, help='Maximum value of predicted depth')
    # Weight settings
    parser.add_argument('--weight_initializer',
        type=str, default='xavier_normal', help='Initialization for weights')
    parser.add_argument('--activation_func',
        type=str, default='leaky_relu', help='Activation function after each layer')
    # Evaluation settings
    parser.add_argument('--min_evaluate_depth',
        type=float, default=0.0, help='Minimum value of depth to evaluate')
    parser.add_argument('--max_evaluate_depth',
        type=float, default=100.0, help='Maximum value of depth to evaluate')
    
    
    parser.add_argument('--save_outputs',
        action='store_true', help='If set then store inputs and outputs into output path')
    parser.add_argument('--keep_input_filenames',
        action='store_true', help='If set then keep original input filenames')
    parser.add_argument('--pretrained_model_path',
        type=str, default=settings.RESTORE_PATH, help='Path to restore depth model from checkpoint')
    
    # Hardware settings
    parser.add_argument('--device',
        type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')
    
    args = parser.parse_args()
    return args



def run(image_path,
        sparse_depth_path,
        intrinsics_path,
        ground_truth_path=None,
        # Input settings
        input_channels_image=settings.INPUT_CHANNELS_IMAGE,
        input_channels_depth=settings.INPUT_CHANNELS_DEPTH,
        normalized_image_range=settings.NORMALIZED_IMAGE_RANGE,
        outlier_removal_kernel_size=settings.OUTLIER_REMOVAL_KERNEL_SIZE,
        outlier_removal_threshold=settings.OUTLIER_REMOVAL_THRESHOLD,
        # Sparse to dense pool settings
        min_pool_sizes_sparse_to_dense_pool=settings.MIN_POOL_SIZES_SPARSE_TO_DENSE_POOL,
        max_pool_sizes_sparse_to_dense_pool=settings.MAX_POOL_SIZES_SPARSE_TO_DENSE_POOL,
        n_convolution_sparse_to_dense_pool=settings.N_CONVOLUTION_SPARSE_TO_DENSE_POOL,
        n_filter_sparse_to_dense_pool=settings.N_FILTER_SPARSE_TO_DENSE_POOL,
        # Depth network settings
        n_filters_encoder_image=settings.N_FILTERS_ENCODER_IMAGE,
        n_filters_encoder_depth=settings.N_FILTERS_ENCODER_DEPTH,
        resolutions_backprojection=settings.RESOLUTIONS_BACKPROJECTION,
        n_filters_decoder=settings.N_FILTERS_DECODER,
        deconv_type=settings.DECONV_TYPE,
        min_predict_depth=settings.MIN_PREDICT_DEPTH,
        max_predict_depth=settings.MAX_PREDICT_DEPTH,
        # Weight settings
        weight_initializer=settings.WEIGHT_INITIALIZER,
        activation_func=settings.ACTIVATION_FUNC,
        # Evaluation settings
        min_evaluate_depth=settings.MIN_EVALUATE_DEPTH,
        max_evaluate_depth=settings.MAX_EVALUATE_DEPTH,
        # Checkpoint settings
        checkpoint_path=settings.CHECKPOINT_PATH,
        depth_model_restore_path=settings.RESTORE_PATH,
        # Output settings
        save_outputs=False,
        keep_input_filenames=False,
        # Hardware settings
        device=settings.DEVICE):

    # Set up output path
    if device == settings.CUDA or device == settings.GPU:
        device = torch.device(settings.CUDA)
    else:
        device = torch.device(settings.CPU)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and output paths
    log_path = os.path.join(checkpoint_path, 'results.txt')
    output_path = os.path.join(checkpoint_path, 'outputs')

    '''
    Load input paths and set up dataloader
    '''
    image_paths = [image_path]# data_utils.read_paths(image_path)
    sparse_depth_paths = [sparse_depth_path]#data_utils.read_paths(sparse_depth_path)
    intrinsics_paths = [intrinsics_path]#data_utils.read_paths(intrinsics_path)

    ground_truth_available = False

    if ground_truth_path != '':
        ground_truth_available = True
        ground_truth_paths = data_utils.read_paths(ground_truth_path)

    n_sample = len(image_paths)

    input_paths = [
        image_paths,
        sparse_depth_paths,
        intrinsics_paths
    ]

    if ground_truth_available:
        input_paths.append(ground_truth_paths)

    for paths in input_paths:
        assert n_sample == len(paths)

    if ground_truth_available:

        ground_truths = []
        for path in ground_truth_paths:
            ground_truth, validity_map = data_utils.load_depth_with_validity_map(path)
            ground_truths.append(np.stack([ground_truth, validity_map], axis=-1))
    else:
        ground_truths = [None] * n_sample

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(
        datasets.KBNetInferenceDataset(
            image_paths=image_paths,
            sparse_depth_paths=sparse_depth_paths,
            intrinsics_paths=intrinsics_paths, use_image_triplet=False),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    # Initialize transforms to normalize image and outlier removal for sparse depth
    transforms = Transforms(
        normalized_image_range=normalized_image_range)

    outlier_removal = OutlierRemoval(
        kernel_size=outlier_removal_kernel_size,
        threshold=outlier_removal_threshold)

    '''
    Set up the model
    '''
    depth_model = KBNetModel(
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        min_pool_sizes_sparse_to_dense_pool=min_pool_sizes_sparse_to_dense_pool,
        max_pool_sizes_sparse_to_dense_pool=max_pool_sizes_sparse_to_dense_pool,
        n_convolution_sparse_to_dense_pool=n_convolution_sparse_to_dense_pool,
        n_filter_sparse_to_dense_pool=n_filter_sparse_to_dense_pool,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        resolutions_backprojection=resolutions_backprojection,
        n_filters_decoder=n_filters_decoder,
        deconv_type=deconv_type,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    # Restore model and set to evaluation mode
    depth_model.restore_model(depth_model_restore_path)
    depth_model.eval()

    parameters_depth_model = depth_model.parameters()

    '''
    Log input paths
    '''
    log('Input paths:', log_path)
    input_paths = [
        image_path,
        sparse_depth_path,
        intrinsics_path,
    ]

    if ground_truth_available:
        input_paths.append(ground_truth_path)

    for path in input_paths:
        log(path, log_path)
    log('', log_path)

    '''
    Log all settings
    '''
    log_input_settings(
        log_path,
        # Input settings
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        normalized_image_range=normalized_image_range,
        outlier_removal_kernel_size=outlier_removal_kernel_size,
        outlier_removal_threshold=outlier_removal_threshold)

    log_network_settings(
        log_path,
        # Sparse to dense pool settings
        min_pool_sizes_sparse_to_dense_pool=min_pool_sizes_sparse_to_dense_pool,
        max_pool_sizes_sparse_to_dense_pool=max_pool_sizes_sparse_to_dense_pool,
        n_convolution_sparse_to_dense_pool=n_convolution_sparse_to_dense_pool,
        n_filter_sparse_to_dense_pool=n_filter_sparse_to_dense_pool,
        # Depth network settings
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        resolutions_backprojection=resolutions_backprojection,
        n_filters_decoder=n_filters_decoder,
        deconv_type=deconv_type,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_depth_model=parameters_depth_model)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        depth_model_restore_path=depth_model_restore_path,
        # Hardware settings
        device=device,
        n_thread=1)

    '''
    Run model
    '''
    # Set up metrics in case groundtruth is available
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    images = []
    output_depths = []
    sparse_depths = []

    time_elapse = 0.0

    for idx, (inputs, ground_truth) in enumerate(zip(dataloader, ground_truths)):

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        image, sparse_depth, intrinsics = inputs

        time_start = time.time()

        # Validity map is where sparse depth is available
        validity_map_depth = torch.where(
            sparse_depth > 0,
            torch.ones_like(sparse_depth),
            sparse_depth)

        # Remove outlier points and update sparse depth and validity map
        filtered_sparse_depth, \
            filtered_validity_map_depth = outlier_removal.remove_outliers(
                sparse_depth=sparse_depth,
                validity_map=validity_map_depth)

        [image] = transforms.transform(
            images_arr=[image],
            random_transform_probability=0.0)

        # Forward through network
        output_depth = depth_model.forward(
            image=image,
            sparse_depth=sparse_depth,
            validity_map_depth=filtered_validity_map_depth,
            intrinsics=intrinsics)

        time_elapse = time_elapse + (time.time() - time_start)

        # Convert to numpy
        output_depth = np.squeeze(output_depth.detach().cpu().numpy())

        # Save to output
        if save_outputs:
            images.append(np.transpose(np.squeeze(image.cpu().numpy()), (1, 2, 0)))
            sparse_depths.append(np.squeeze(filtered_sparse_depth.cpu().numpy()))
            output_depths.append(output_depth)

        if ground_truth_available:
            ground_truth = np.squeeze(ground_truth)

            validity_map = ground_truth[:, :, 1]
            ground_truth = ground_truth[:, :, 0]

            validity_mask = np.where(validity_map > 0, 1, 0)
            min_max_mask = np.logical_and(
                ground_truth > min_evaluate_depth,
                ground_truth < max_evaluate_depth)
            mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

            output_depth = output_depth[mask]
            ground_truth = ground_truth[mask]

            mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
            rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
            imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
            irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

    # Compute total time elapse in ms
    time_elapse = time_elapse * 1000.0

    if ground_truth_available:
        mae_mean   = np.mean(mae)
        rmse_mean  = np.mean(rmse)
        imae_mean  = np.mean(imae)
        irmse_mean = np.mean(irmse)

        mae_std = np.std(mae)
        rmse_std = np.std(rmse)
        imae_std = np.std(imae)
        irmse_std = np.std(irmse)

        # Print evaluation results to console and file
        log('Evaluation results:', log_path)
        log('{:>8}  {:>8}  {:>8}  {:>8}'.format(
            'MAE', 'RMSE', 'iMAE', 'iRMSE'),
            log_path)
        log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            mae_mean, rmse_mean, imae_mean, irmse_mean),
            log_path)

        log('{:>8}  {:>8}  {:>8}  {:>8}'.format(
            '+/-', '+/-', '+/-', '+/-'),
            log_path)
        log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            mae_std, rmse_std, imae_std, irmse_std),
            log_path)

    # Log run time
    log('Total time: {:.2f} ms  Average time per sample: {:.2f} ms'.format(
        time_elapse, time_elapse / float(n_sample)))

    if save_outputs:
        log('Saving outputs to {}'.format(output_path), log_path)

        outputs = zip(images, output_depths, sparse_depths, ground_truths)

        image_dirpath = os.path.join(output_path, 'image')
        output_depth_dirpath = os.path.join(output_path, 'output_depth')
        sparse_depth_dirpath = os.path.join(output_path, 'sparse_depth')
        ground_truth_dirpath = os.path.join(output_path, 'ground_truth')

        dirpaths = [
            image_dirpath,
            output_depth_dirpath,
            sparse_depth_dirpath,
            ground_truth_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

        for idx, (image, output_depth, sparse_depth, ground_truth) in enumerate(outputs):

            if keep_input_filenames:
                filename = os.path.basename(image_paths[idx])
            else:
                filename = '{:010d}.png'.format(idx)

            image_path = os.path.join(image_dirpath, filename)
            image = (255 * image).astype(np.uint8)
            Image.fromarray(image).save(image_path)

            output_depth_path = os.path.join(output_depth_dirpath, filename)
            data_utils.save_depth(output_depth, output_depth_path)

            sparse_depth_path = os.path.join(sparse_depth_dirpath, filename)
            data_utils.save_depth(sparse_depth, sparse_depth_path)

            if ground_truth_available:
                ground_truth_path = os.path.join(ground_truth_dirpath, filename)
                data_utils.save_depth(ground_truth[..., 0], ground_truth_path)


