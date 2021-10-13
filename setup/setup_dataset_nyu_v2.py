import warnings
warnings.filterwarnings("ignore")

import os, sys, glob, cv2, argparse
import multiprocessing as mp
import numpy as np
sys.path.insert(0, 'src')
import data_utils
from sklearn.cluster import MiniBatchKMeans


N_CLUSTER = 1500
O_HEIGHT = 480
O_WIDTH = 640
N_HEIGHT = 416
N_WIDTH = 576
MIN_POINTS = 1100
TEMPORAL_WINDOW = 21
RANDOM_SEED = 1

parser = argparse.ArgumentParser()

parser.add_argument('--sparse_depth_distro_type', type=str, default='corner')
parser.add_argument('--n_points',                 type=int, default=N_CLUSTER)
parser.add_argument('--min_points',               type=int, default=MIN_POINTS)
parser.add_argument('--temporal_window',          type=int, default=TEMPORAL_WINDOW)
parser.add_argument('--n_height',                 type=int, default=N_HEIGHT)
parser.add_argument('--n_width',                  type=int, default=N_WIDTH)

args = parser.parse_args()


NYU_ROOT_DIRPATH = \
    os.path.join('data', 'nyu_v2')
NYU_OUTPUT_DIRPATH = \
    os.path.join('data', 'nyu_v2_adverse_weather')

NYU_TEST_IMAGE_SPLIT_FILEPATH = \
    os.path.join('setup', 'nyu_v2_test_image.txt')
NYU_TEST_DEPTH_SPLIT_FILEPATH = \
    os.path.join('setup', 'nyu_v2_test_depth.txt')

TRAIN_REF_DIRPATH = os.path.join('training', 'nyu_v2')
VAL_REF_DIRPATH = os.path.join('validation', 'nyu_v2')
TEST_REF_DIRPATH = os.path.join('testing', 'nyu_v2')

TRAIN_IMAGE_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_REF_DIRPATH, 'nyu_v2_train_image_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_REF_DIRPATH, 'nyu_v2_train_sparse_depth_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_INTERP_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_REF_DIRPATH, 'nyu_v2_train_interp_depth_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_REF_DIRPATH, 'nyu_v2_train_validity_map_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_REF_DIRPATH, 'nyu_v2_train_ground_truth_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_REF_DIRPATH, 'nyu_v2_train_intrinsics_{}.txt'.format(args.sparse_depth_distro_type))

VAL_IMAGE_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_image_{}.txt'.format(args.sparse_depth_distro_type))
VAL_SPARSE_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_sparse_depth_{}.txt'.format(args.sparse_depth_distro_type))
VAL_INTERP_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_interp_depth_{}.txt'.format(args.sparse_depth_distro_type))
VAL_VALIDITY_MAP_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_validity_map_{}.txt'.format(args.sparse_depth_distro_type))
VAL_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_ground_truth_{}.txt'.format(args.sparse_depth_distro_type))
VAL_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_intrinsics_{}.txt'.format(args.sparse_depth_distro_type))

TEST_IMAGE_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_image_{}.txt'.format(args.sparse_depth_distro_type))
TEST_SPARSE_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_sparse_depth_{}.txt'.format(args.sparse_depth_distro_type))
TEST_INTERP_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_interp_depth_{}.txt'.format(args.sparse_depth_distro_type))
TEST_VALIDITY_MAP_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_validity_map_{}.txt'.format(args.sparse_depth_distro_type))
TEST_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_ground_truth_{}.txt'.format(args.sparse_depth_distro_type))
TEST_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_intrinsics_{}.txt'.format(args.sparse_depth_distro_type))


def process_frame(inputs):

    image0_path, image1_path, image2_path, ground_truth_path = inputs

    # Load image (for corner detection) to generate valid map
    image0 = cv2.imread(image0_path)
    image0 = np.float32(cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY))

    # Load dense depth
    ground_truth = data_utils.load_depth(ground_truth_path)

    assert image0.shape[0] == ground_truth.shape[0] and image0.shape[1] == ground_truth.shape[1]
    assert image0.shape[0] == O_HEIGHT and image0.shape[1] == O_WIDTH

    # Crop away white borders
    if args.n_height != O_HEIGHT or args.n_width != O_WIDTH:
        d_height = O_HEIGHT - args.n_height
        d_width = O_WIDTH - args.n_width

        y_start = d_height // 2
        x_start = d_width // 2
        y_end = y_start + args.n_height
        x_end = x_start + args.n_width

        image0 = image0[y_start:y_end, x_start:x_end]
        ground_truth = ground_truth[y_start:y_end, x_start:x_end]

    if args.sparse_depth_distro_type == 'corner':
        N_INIT_CORNER = 30000

        # Run Harris corner detector
        corners = cv2.cornerHarris(image0, blockSize=5, ksize=3, k=0.04)

        # Remove the corners that are located on invalid depth locations
        corners = corners * np.where(ground_truth > 0.0, 1.0, 0.0)

        # Vectorize corner map to 1D vector and select N_INIT_CORNER corner locations
        corners = corners.ravel()
        corner_locations = np.argsort(corners)[0:N_INIT_CORNER]

        # Get locations of corners as indices as (x, y)
        corner_locations = np.unravel_index(
            corner_locations,
            (image0.shape[0], image0.shape[1]))

        # Convert to (y, x) convention
        corner_locations = \
            np.transpose(np.array([corner_locations[0], corner_locations[1]]))

        # Cluster them into n_points (number of output points)
        kmeans = MiniBatchKMeans(
            n_clusters=args.n_points,
            max_iter=2,
            n_init=1,
            init_size=None,
            random_state=RANDOM_SEED,
            reassignment_ratio=1e-11)
        kmeans.fit(corner_locations)

        # Use k-Means means as corners
        selected_indices = kmeans.cluster_centers_.astype(np.uint16)

    elif args.sparse_depth_distro_type == 'uniform':
        indices = \
            np.array([[h, w] for h in range(args.n_height) for w in range(args.n_width)])

        # Randomly select n_points number of points
        selected_indices = \
            np.random.permutation(range(args.n_height * args.n_width))[0:args.n_points]
        selected_indices = indices[selected_indices]

    else:
        raise ValueError('Unsupported sparse depth distribution type: {}'.format(
            args.sparse_depth_distro_type))

    # Convert the indicies into validity map
    validity_map = np.zeros_like(image0).astype(np.int16)
    validity_map[selected_indices[:, 0], selected_indices[:, 1]] = 1.0

    # Build validity map from selected points, keep only ones greater than 0
    validity_map = np.where(validity_map * ground_truth > 0.0, 1.0, 0.0)

    # Get sparse depth based on validity map
    sparse_depth = validity_map * ground_truth

    # Shape check
    error_flag = False

    if np.squeeze(sparse_depth).shape != (args.n_height, args.n_width):
        error_flag = True
        print('FAILED: np.squeeze(sparse_depth).shape != ({}, {})'.format(args.n_height, args.n_width))

    # Validity map check
    if not np.array_equal(np.unique(validity_map), np.array([0, 1])):
        error_flag = True
        print('FAILED: not np.array_equal(np.unique(validity_map), np.array([0, 1]))')

    if validity_map.sum() < args.min_points:
        error_flag = True
        print('FAILED: validity_map.sum() < MIN_POINTS')

    # Depth value check
    if np.min(ground_truth) < 0.0 or np.max(ground_truth) > 256.0:
        error_flag = True
        print('FAILED: np.min(ground_truth) < 0.0 or np.max(ground_truth) > 256.0')

    if np.sum(np.where(validity_map > 0.0, 1.0, 0.0)) < args.min_points:
        error_flag = True
        print('FAILED: np.sum(np.where(validity_map > 0.0, 1.0, 0.0)) < MIN_POINTS', np.sum(np.where(validity_map > 0.0, 1.0, 0.0)))

    if np.sum(np.where(ground_truth > 0.0, 1.0, 0.0)) < args.min_points:
        error_flag = True
        print('FAILED: np.sum(np.where(ground_truth > 0.0, 1.0, 0.0)) < MIN_POINTS')

    # NaN check
    if np.any(np.isnan(sparse_depth)):
        error_flag = True
        print('FAILED: np.any(np.isnan(sparse_depth))')

    if not error_flag:

        # Read images and concatenate together
        image0 = cv2.imread(image0_path)
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        if args.n_height != O_HEIGHT or args.n_width != O_WIDTH:
            image0 = image0[y_start:y_end, x_start:x_end, :]
            image1 = image1[y_start:y_end, x_start:x_end, :]
            image2 = image2[y_start:y_end, x_start:x_end, :]

        imagec = np.concatenate([image1, image0, image2], axis=1)

        interp_depth = data_utils.interpolate_depth(sparse_depth, validity_map)

        # Example: nyu/training/depths/raw_data/bedroom_0001/r-1294886360.208451-2996770081.png
        image_output_path = image0_path \
            .replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH)
        sparse_depth_output_path = ground_truth_path \
            .replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH) \
            .replace('depth', 'sparse_depth')
        interp_depth_output_path = ground_truth_path \
            .replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH) \
            .replace('depth', 'interp_depth')
        validity_map_output_path = ground_truth_path \
            .replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH) \
            .replace('depth', 'validity_map')
        ground_truth_output_path = ground_truth_path \
            .replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH) \
            .replace('depth', 'ground_truth')

        image_output_dirpath = os.path.dirname(image_output_path)
        sparse_depth_output_dirpath = os.path.dirname(sparse_depth_output_path)
        interp_depth_output_dirpath = os.path.dirname(interp_depth_output_path)
        validity_map_output_dirpath = os.path.dirname(validity_map_output_path)
        ground_truth_output_dirpath = os.path.dirname(ground_truth_output_path)

        # Create output directories
        output_dirpaths = [
            image_output_dirpath,
            sparse_depth_output_dirpath,
            interp_depth_output_dirpath,
            validity_map_output_dirpath,
            ground_truth_output_dirpath,
        ]

        for dirpath in output_dirpaths:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)

        # Write to file
        cv2.imwrite(image_output_path, imagec)
        data_utils.save_depth(sparse_depth, sparse_depth_output_path)
        data_utils.save_depth(interp_depth, interp_depth_output_path)
        data_utils.save_validity_map(validity_map, validity_map_output_path)
        data_utils.save_depth(ground_truth, ground_truth_output_path)
    else:
        print('Found error in {}'.format(ground_truth_path))
        image_output_path = 'error'
        sparse_depth_output_path = 'error'
        interp_depth_output_path = 'error'
        validity_map_output_path = 'error'
        ground_truth_output_path = 'error'

    return (image_output_path,
            sparse_depth_output_path,
            interp_depth_output_path,
            validity_map_output_path,
            ground_truth_output_path)

def filter_sequence(seq):
    keep_sequence = \
        '_0000/' in seq or \
        '_0001/' in seq or \
        '_0002/' in seq or \
        '_0003/' in seq or \
        '_0004/' in seq

    return keep_sequence

def filter_paths(paths):
    paths_ = []

    for path in paths:
        if filter_sequence(path):
            paths_.append(path)

    return paths_


# Create output directories first
dirpaths = [
    NYU_OUTPUT_DIRPATH,
    TRAIN_REF_DIRPATH,
    VAL_REF_DIRPATH,
    TEST_REF_DIRPATH
]

for dirpath in dirpaths:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


'''
Setup intrinsics (values are copied from camera_params.m)
'''
fx_rgb = 518.85790117450188
fy_rgb = 519.46961112127485
cx_rgb = 325.58244941119034
cy_rgb = 253.73616633400465
intrinsic_matrix = np.array([
    [fx_rgb,   0.,     cx_rgb],
    [0.,       fy_rgb, cy_rgb],
    [0.,       0.,     1.    ]], dtype=np.float32)


if args.n_height != O_HEIGHT or args.n_width != O_WIDTH:

    d_height = O_HEIGHT - args.n_height
    d_width = O_WIDTH - args.n_width

    y_start = d_height // 2
    x_start = d_width // 2

    intrinsic_matrix = intrinsic_matrix + [[0.0, 0.0, -x_start],
                                           [0.0, 0.0, -y_start],
                                           [0.0, 0.0, 0.0     ]]

intrinsics_output_path = os.path.join(NYU_OUTPUT_DIRPATH, 'intrinsics.npy')
np.save(intrinsics_output_path, intrinsic_matrix)


'''
Process training paths
'''
train_image_output_paths = []
train_sparse_depth_output_paths = []
train_interp_depth_output_paths = []
train_validity_map_output_paths = []
train_ground_truth_output_paths = []
train_intrinsics_output_paths = [intrinsics_output_path]

train_image_sequences = sorted(glob.glob(
    os.path.join(NYU_ROOT_DIRPATH, 'training', 'images', 'raw_data', '*/')))
train_depth_sequences = sorted(glob.glob(
    os.path.join(NYU_ROOT_DIRPATH, 'training', 'depths', 'raw_data', '*/')))

# Use only a subset for training
train_image_sequences = filter_paths(train_image_sequences)
train_depth_sequences = filter_paths(train_depth_sequences)

w = int(args.temporal_window // 2)

for image_sequence, depth_sequence in zip(train_image_sequences, train_depth_sequences):

    # Fetch image and dense depth from sequence directory
    image_paths = \
        sorted(glob.glob(os.path.join(image_sequence, '*.png')))
    ground_truth_paths = \
        sorted(glob.glob(os.path.join(depth_sequence, '*.png')))

    n_sample = len(image_paths)

    for image_path, ground_truth_path in zip(image_paths, ground_truth_paths):
        assert os.path.join(*(image_path.split(os.sep)[-3:])) == os.path.join(*(image_path.split(os.sep)[-3:]))

    pool_input = [
        (image_paths[idx], image_paths[idx-w], image_paths[idx+w], ground_truth_paths[idx])
        for idx in range(w, n_sample - w)
    ]

    print('Processing {} samples in: {}'.format(n_sample - 2 * w + 1, image_sequence))

    with mp.Pool() as pool:
        pool_results = pool.map(process_frame, pool_input)

        for result in pool_results:
            image_output_path, \
                sparse_depth_output_path, \
                interp_depth_output_path, \
                validity_map_output_path, \
                ground_truth_output_path = result

            error_encountered = \
                image_output_path == 'error' or \
                sparse_depth_output_path == 'error' or \
                interp_depth_output_path == 'error' or \
                validity_map_output_path == 'error' or \
                ground_truth_output_path == 'error'

            if error_encountered:
                continue

            # Collect filepaths
            train_image_output_paths.append(image_output_path)
            train_sparse_depth_output_paths.append(sparse_depth_output_path)
            train_interp_depth_output_paths.append(interp_depth_output_path)
            train_validity_map_output_paths.append(validity_map_output_path)
            train_ground_truth_output_paths.append(ground_truth_output_path)

train_intrinsics_output_paths = train_intrinsics_output_paths * len(train_image_output_paths)

print('Storing %d training image file paths into: %s' %
    (len(train_image_output_paths), TRAIN_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_IMAGE_OUTPUT_FILEPATH, train_image_output_paths)

print('Storing %d training sparse depth file paths into: %s' %
    (len(train_sparse_depth_output_paths), TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH, train_sparse_depth_output_paths)

print('Storing %d training interp depth file paths into: %s' %
    (len(train_interp_depth_output_paths), TRAIN_INTERP_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_INTERP_DEPTH_OUTPUT_FILEPATH, train_interp_depth_output_paths)

print('Storing %d training validity_map file paths into: %s' %
    (len(train_validity_map_output_paths), TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH, train_validity_map_output_paths)

print('Storing %d training ground truth file paths into: %s' %
    (len(train_ground_truth_output_paths), TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH, train_ground_truth_output_paths)

print('Storing %d training intrinsics file paths into: %s' %
    (len(train_intrinsics_output_paths), TRAIN_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_INTRINSICS_OUTPUT_FILEPATH, train_intrinsics_output_paths)


'''
Process validation and testing paths
'''
test_image_split_paths = data_utils.read_paths(NYU_TEST_IMAGE_SPLIT_FILEPATH)

val_image_output_paths = []
val_sparse_depth_output_paths = []
val_interp_depth_output_paths = []
val_validity_map_output_paths = []
val_ground_truth_output_paths = []
val_intrinsics_output_paths = [intrinsics_output_path]

test_image_output_paths = []
test_sparse_depth_output_paths = []
test_interp_depth_output_paths = []
test_validity_map_output_paths = []
test_ground_truth_output_paths = []
test_intrinsics_output_paths = [intrinsics_output_path]

test_image_paths = sorted(glob.glob(
    os.path.join(NYU_ROOT_DIRPATH, 'testing', 'images', '*.png')))
test_ground_truth_paths = sorted(glob.glob(
    os.path.join(NYU_ROOT_DIRPATH, 'testing', 'depths', '*.png')))

n_sample = len(test_image_paths)

for image_path, ground_truth_path in zip(test_image_paths, test_ground_truth_paths):
    assert os.path.join(*(image_path.split(os.sep)[-3:])) == os.path.join(*(image_path.split(os.sep)[-3:]))

pool_input = [
    (test_image_paths[idx], test_image_paths[idx], test_image_paths[idx], test_ground_truth_paths[idx])
    for idx in range(n_sample)
]

print('Processing {} samples for validation and testing'.format(n_sample))

with mp.Pool() as pool:
    pool_results = pool.map(process_frame, pool_input)

    for result in pool_results:
        image_output_path, \
            sparse_depth_output_path, \
            interp_depth_output_path, \
            validity_map_output_path, \
            ground_truth_output_path = result

        error_encountered = \
            image_output_path == 'error' or \
            sparse_depth_output_path == 'error' or \
            interp_depth_output_path == 'error' or \
            validity_map_output_path == 'error' or \
            ground_truth_output_path == 'error'

        if error_encountered:
            continue

        test_split = False
        for test_image_path in test_image_split_paths:
            if test_image_path in image_output_path:
                test_split = True

        if test_split:
            # Collect test filepaths
            test_image_output_paths.append(image_output_path)
            test_sparse_depth_output_paths.append(sparse_depth_output_path)
            test_interp_depth_output_paths.append(interp_depth_output_path)
            test_validity_map_output_paths.append(validity_map_output_path)
            test_ground_truth_output_paths.append(ground_truth_output_path)
        else:
            # Collect validation filepaths
            val_image_output_paths.append(image_output_path)
            val_sparse_depth_output_paths.append(sparse_depth_output_path)
            val_interp_depth_output_paths.append(interp_depth_output_path)
            val_validity_map_output_paths.append(validity_map_output_path)
            val_ground_truth_output_paths.append(ground_truth_output_path)

val_intrinsics_output_paths = val_intrinsics_output_paths * len(val_image_output_paths)
test_intrinsics_output_paths = test_intrinsics_output_paths * len(test_image_output_paths)

'''
Write validation output paths
'''
print('Storing %d validation image file paths into: %s' %
    (len(val_image_output_paths), VAL_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_IMAGE_OUTPUT_FILEPATH, val_image_output_paths)

print('Storing %d validation sparse depth file paths into: %s' %
    (len(val_sparse_depth_output_paths), VAL_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_SPARSE_DEPTH_OUTPUT_FILEPATH, val_sparse_depth_output_paths)

print('Storing %d validation interp depth file paths into: %s' %
    (len(val_interp_depth_output_paths), VAL_INTERP_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_INTERP_DEPTH_OUTPUT_FILEPATH, val_interp_depth_output_paths)

print('Storing %d validation validity_map file paths into: %s' %
    (len(val_validity_map_output_paths), VAL_VALIDITY_MAP_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_VALIDITY_MAP_OUTPUT_FILEPATH, val_validity_map_output_paths)

print('Storing %d validation dense depth file paths into: %s' %
    (len(val_ground_truth_output_paths), VAL_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_GROUND_TRUTH_OUTPUT_FILEPATH, val_ground_truth_output_paths)

print('Storing %d validation intrinsics file paths into: %s' %
    (len(val_intrinsics_output_paths), VAL_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_INTRINSICS_OUTPUT_FILEPATH, val_intrinsics_output_paths)


'''
Write testing output paths
'''
print('Storing %d testing image file paths into: %s' %
    (len(test_image_output_paths), TEST_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_IMAGE_OUTPUT_FILEPATH, test_image_output_paths)

print('Storing %d testing sparse depth file paths into: %s' %
    (len(test_sparse_depth_output_paths), TEST_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_SPARSE_DEPTH_OUTPUT_FILEPATH, test_sparse_depth_output_paths)

print('Storing %d testing interp depth file paths into: %s' %
    (len(test_interp_depth_output_paths), TEST_INTERP_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_INTERP_DEPTH_OUTPUT_FILEPATH, test_interp_depth_output_paths)

print('Storing %d testing validity_map file paths into: %s' %
    (len(test_validity_map_output_paths), TEST_VALIDITY_MAP_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_VALIDITY_MAP_OUTPUT_FILEPATH, test_validity_map_output_paths)

print('Storing %d testing dense depth file paths into: %s' %
    (len(test_ground_truth_output_paths), TEST_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_GROUND_TRUTH_OUTPUT_FILEPATH, test_ground_truth_output_paths)

print('Storing %d testing intrinsics file paths into: %s' %
    (len(test_intrinsics_output_paths), TEST_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_INTRINSICS_OUTPUT_FILEPATH, test_intrinsics_output_paths)
