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
import os, sys, glob, argparse, cv2
import multiprocessing as mp
import numpy as np
sys.path.insert(0, 'src')
import data_utils


'''
Paths for KITTI dataset
'''
KITTI_RAW_DATA_DIRPATH = os.path.join('data', 'kitti_raw_data')
KITTI_DEPTH_COMPLETION_DIRPATH = os.path.join('data', 'kitti_depth_completion')

KITTI_TRAINVAL_SPARSE_DEPTH_DIRPATH = os.path.join(
    KITTI_DEPTH_COMPLETION_DIRPATH, 'train_val_split', 'sparse_depth')
KITTI_TRAINVAL_GROUND_TRUTH_DIRPATH = os.path.join(
    KITTI_DEPTH_COMPLETION_DIRPATH, 'train_val_split', 'ground_truth')
KITTI_VALIDATION_DIRPATH = os.path.join(
    KITTI_DEPTH_COMPLETION_DIRPATH, 'validation')
KITTI_TESTING_DIRPATH = os.path.join(
    KITTI_DEPTH_COMPLETION_DIRPATH, 'testing')
KITTI_CALIBRATION_FILENAME = 'calib_cam_to_cam.txt'

KITTI_STATIC_FRAMES_FILEPATH = os.path.join('setup', 'kitti_static_frames.txt')

# To be concatenated to sequence path
KITTI_TRAINVAL_IMAGE_REFPATH = os.path.join('proj_depth', 'velodyne_raw')
KITTI_TRAINVAL_SPARSE_DEPTH_REFPATH = os.path.join('proj_depth', 'velodyne_raw')
KITTI_TRAINVAL_GROUND_TRUTH_REFPATH = os.path.join('proj_depth', 'groundtruth')


'''
Output paths
'''
KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH = os.path.join(
    'data', 'kitti_depth_completion_kbnet')

TRAIN_OUTPUT_REF_DIRPATH = os.path.join('training', 'kitti')
VAL_OUTPUT_REF_DIRPATH = os.path.join('validation', 'kitti')
TEST_OUTPUT_REF_DIRPATH = os.path.join('testing', 'kitti')

TRAIN_IMAGE_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_image.txt')
TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_sparse_depth.txt')
TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_validity_map.txt')
TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_ground_truth.txt')
TRAIN_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_intrinsics.txt')

TRAIN_IMAGE_CLEAN_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_image-clean.txt')
TRAIN_SPARSE_DEPTH_CLEAN_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_sparse_depth-clean.txt')
TRAIN_VALIDITY_MAP_CLEAN_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_validity_map-clean.txt')
TRAIN_GROUND_TRUTH_CLEAN_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_ground_truth-clean.txt')
TRAIN_INTRINSICS_CLEAN_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_intrinsics-clean.txt')

UNUSED_IMAGE_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_image.txt')
UNUSED_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_sparse_depth.txt')
UNUSED_VALIDITY_MAP_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_validity_map.txt')
UNUSED_GROUND_TRUTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_ground_truth.txt')
UNUSED_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_intrinsics_depth.txt')

VAL_IMAGE_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH, 'kitti_val_image.txt')
VAL_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH, 'kitti_val_sparse_depth.txt')
VAL_VALIDITY_MAP_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH, 'kitti_val_validity_map.txt')
VAL_GROUND_TRUTH_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH, 'kitti_val_ground_truth.txt')
VAL_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH, 'kitti_val_intrinsics.txt')

TEST_IMAGE_OUTPUT_FILEPATH = os.path.join(
    TEST_OUTPUT_REF_DIRPATH, 'kitti_test_image.txt')
TEST_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(
    TEST_OUTPUT_REF_DIRPATH, 'kitti_test_sparse_depth.txt')
TEST_VALIDITY_MAP_OUTPUT_FILEPATH = os.path.join(
    TEST_OUTPUT_REF_DIRPATH, 'kitti_test_validity_map.txt')
TEST_GROUND_TRUTH_OUTPUT_FILEPATH = os.path.join(
    TEST_OUTPUT_REF_DIRPATH, 'kitti_test_ground_truth.txt')
TEST_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TEST_OUTPUT_REF_DIRPATH, 'kitti_test_intrinsics.txt')


parser = argparse.ArgumentParser()

parser.add_argument('--paths_only', action='store_true')
parser.add_argument('--n_thread',  type=int, default=8)
args = parser.parse_args()


def process_frame(inputs):
    '''
    Processes a single frame

    Arg(s):
        inputs : tuple[str]
            image path at time t=0,
            image path at time t=1,
            image path at time t=-1,
            sparse depth path at time t=0,
            ground truth path at time t=0,
            boolean flag if set then create paths only
    Returns:
        str : output concatenated image path at time t=0
        str : output sparse depth path at time t=0
        str : output validity map path at time t=0
        str : output ground truth path at time t=0
    '''

    image0_path, \
        image1_path, \
        image2_path, \
        sparse_depth_path, \
        ground_truth_path, \
        paths_only = inputs

    if not paths_only:
        # Read images and concatenate together
        image0 = cv2.imread(image0_path)
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        image = np.concatenate([image1, image0, image2], axis=1)

        _, validity_map = data_utils.load_depth_with_validity_map(sparse_depth_path)

    # Create validity map and image output path
    validity_map_output_path = sparse_depth_path \
        .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH) \
        .replace('sparse_depth', 'validity_map')
    image_output_path = validity_map_output_path \
        .replace(os.path.join(os.sep + 'proj_depth', 'velodyne_raw'), '') \
        .replace('validity_map', 'image')

    # Create output directories
    for output_path in [image_output_path, validity_map_output_path]:
        output_dirpath = os.path.dirname(output_path)
        if not os.path.exists(output_dirpath):
            try:
                os.makedirs(output_dirpath)
            except FileExistsError:
                pass

    if not paths_only:
        # Write to disk
        data_utils.save_validity_map(validity_map, validity_map_output_path)
        cv2.imwrite(image_output_path, image)

    return (image_output_path,
            sparse_depth_path,
            validity_map_output_path,
            ground_truth_path)


for dirpath in [TRAIN_OUTPUT_REF_DIRPATH, VAL_OUTPUT_REF_DIRPATH, TEST_OUTPUT_REF_DIRPATH]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Build a mapping between the camera intrinsics to the directories
intrinsics_files = sorted(glob.glob(os.path.join(
    KITTI_RAW_DATA_DIRPATH, '*', KITTI_CALIBRATION_FILENAME)))
intrinsics_dkeys = {}

for intrinsics_file in intrinsics_files:
    # Example: data/kitti_depth_completion_voiced/data/2011_09_26/kin2.npy
    intrinsics2_path = intrinsics_file \
        .replace(KITTI_RAW_DATA_DIRPATH, os.path.join(KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH, 'data')) \
        .replace(KITTI_CALIBRATION_FILENAME, 'intrinsics2.npy')
    intrinsics3_path = intrinsics_file \
        .replace(KITTI_RAW_DATA_DIRPATH, os.path.join(KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH, 'data')) \
        .replace(KITTI_CALIBRATION_FILENAME, 'intrinsics3.npy')

    sequence_dirpath = os.path.split(intrinsics2_path)[0]
    if not os.path.exists(sequence_dirpath):
        os.makedirs(sequence_dirpath)

    if not args.paths_only:
        calib = data_utils.load_calibration(intrinsics_file)
        intrinsics2 = np.reshape(calib['P_rect_02'], [3, 4])
        intrinsics2 = intrinsics2[:3, :3].astype(np.float32)
        intrinsics3 = np.reshape(calib['P_rect_03'], [3, 4])
        intrinsics3 = intrinsics3[:3, :3].astype(np.float32)

        # Store as numpy
        np.save(intrinsics2_path, intrinsics2)
        np.save(intrinsics3_path, intrinsics3)

    # Add as keys to instrinsics dictionary
    sequence_date = intrinsics_file.split(os.sep)[2]
    intrinsics_dkeys[(sequence_date, 'image_02')] = intrinsics2_path
    intrinsics_dkeys[(sequence_date, 'image_03')] = intrinsics3_path


'''
Create validity maps and paths for sparse depth and ground truth for training
'''
train_image_output_paths = []
train_sparse_depth_output_paths = []
train_validity_map_output_paths = []
train_ground_truth_output_paths = []
train_intrinsics_output_paths = []
unused_image_output_paths = []
unused_sparse_depth_output_paths = []
unused_validity_map_output_paths = []
unused_ground_truth_output_paths = []
unused_intrinsics_output_paths = []

# Iterate through train and val directories
for refdir in ['train', 'val']:
    sparse_depth_sequence_dirpath = glob.glob(
        os.path.join(KITTI_TRAINVAL_SPARSE_DEPTH_DIRPATH, refdir, '*/'))

    # Iterate through sequences
    for sequence_dirpath in sparse_depth_sequence_dirpath:

        # Iterate through cameras 02 and 03
        for camera_dirpath in ['image_02', 'image_03']:
            # Contruct sparse depth dirpaths
            sparse_depth_paths = sorted(glob.glob(
                os.path.join(
                    sequence_dirpath, KITTI_TRAINVAL_SPARSE_DEPTH_REFPATH, camera_dirpath, '*.png')))

            # Construct ground truth diraths
            ground_truth_sequence_dirpath = sequence_dirpath.replace(
                'sparse_depth', 'ground_truth')
            ground_truth_paths = sorted(glob.glob(
                os.path.join(
                    ground_truth_sequence_dirpath, KITTI_TRAINVAL_GROUND_TRUTH_REFPATH, camera_dirpath, '*.png')))

            assert len(sparse_depth_paths) == len(ground_truth_paths)

            # Obtain sequence dirpath in raw data
            sequence = sparse_depth_paths[0].split(os.sep)[5]
            sequence_date = sequence[0:10]
            raw_sequence_dirpath = os.path.join(
                KITTI_RAW_DATA_DIRPATH, sequence_date, sequence, camera_dirpath, 'data')
            image_paths = sorted(
                glob.glob(os.path.join(raw_sequence_dirpath, '*.png')))

            intrinsics_output_path = intrinsics_dkeys[sequence_date, camera_dirpath]

            print('Processing {} samples using KITTI sequence={} camera={}'.format(
                len(sparse_depth_paths), sequence_dirpath.split(os.sep)[-2], camera_dirpath))

            pool_inputs = []
            # Load sparse depth and save validity map
            for idx in range(len(sparse_depth_paths)):
                sparse_depth_path = sparse_depth_paths[idx]
                ground_truth_path = ground_truth_paths[idx]
                filename0 = os.path.split(sparse_depth_paths[idx])[-1]

                assert os.path.split(ground_truth_path)[-1] == filename0

                # Construct image filepaths
                image0_path = os.path.join(raw_sequence_dirpath, filename0)
                image0_path_idx = image_paths.index(image0_path)
                image1_path = image_paths[image0_path_idx-1]
                image2_path = image_paths[image0_path_idx+1]

                pool_inputs.append((
                    image0_path,
                    image1_path,
                    image2_path,
                    sparse_depth_path,
                    ground_truth_path,
                    args.paths_only))

            with mp.Pool(args.n_thread) as pool:
                pool_results = pool.map(process_frame, pool_inputs)

                for result in pool_results:
                    image_output_path, \
                        sparse_depth_path, \
                        validity_map_output_path, \
                        ground_truth_path = result

                    if refdir == 'train':
                        train_image_output_paths.append(image_output_path)
                        train_sparse_depth_output_paths.append(sparse_depth_path)
                        train_validity_map_output_paths.append(validity_map_output_path)
                        train_ground_truth_output_paths.append(ground_truth_path)
                        train_intrinsics_output_paths.append(intrinsics_output_path)
                    elif refdir == 'val':
                        unused_image_output_paths.append(image_output_path)
                        unused_sparse_depth_output_paths.append(sparse_depth_path)
                        unused_validity_map_output_paths.append(validity_map_output_path)
                        unused_ground_truth_output_paths.append(ground_truth_path)
                        unused_intrinsics_output_paths.append(intrinsics_output_path)

            print('Completed processing {} samples using KITTI sequence={} camera={}'.format(
                len(sparse_depth_paths), sequence_dirpath.split(os.sep)[-2], camera_dirpath))

kitti_static_frames_paths = data_utils.read_paths(KITTI_STATIC_FRAMES_FILEPATH)

kitti_static_frames_parts = []
for path in kitti_static_frames_paths:
    parts = path.split(' ')
    kitti_static_frames_parts.append((parts[1], parts[2]))

train_image_clean_output_paths = []
train_sparse_depth_clean_output_paths = []
train_validity_map_clean_output_paths = []
train_ground_truth_clean_output_paths = []
train_intrinsics_clean_output_paths = []

n_removed = 0
n_sample = len(train_image_output_paths)

for idx in range(n_sample):
    image_output_path = train_image_output_paths[idx]
    sparse_depth_output_path = train_sparse_depth_output_paths[idx]
    validity_map_output_path = train_validity_map_output_paths[idx]
    ground_truth_output_path = train_ground_truth_output_paths[idx]
    intrinsics_output_path = train_intrinsics_output_paths[idx]

    filename = os.path.basename(image_output_path)
    assert filename == os.path.basename(sparse_depth_output_path)
    assert filename == os.path.basename(validity_map_output_path)
    assert filename == os.path.basename(ground_truth_output_path)

    # If static path parts are found in path, then mark as static
    is_static = False
    for parts in kitti_static_frames_parts:
        if parts[0] in image_output_path and parts[1] in image_output_path:
            is_static = True
            break

    # Remove from the final set of paths
    if is_static:
        n_removed = n_removed + 1
        continue
    else:
        train_image_clean_output_paths.append(image_output_path)
        train_sparse_depth_clean_output_paths.append(sparse_depth_output_path)
        train_validity_map_clean_output_paths.append(validity_map_output_path)
        train_ground_truth_clean_output_paths.append(ground_truth_output_path)
        train_intrinsics_clean_output_paths.append(intrinsics_output_path)

    sys.stdout.write(
        'Processed {}/{} examples \r'.format(idx + 1, n_sample))
    sys.stdout.flush()


# Write all training file paths
print('Storing training image file paths into: {}'.format(
    TRAIN_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_IMAGE_OUTPUT_FILEPATH, train_image_output_paths)

print('Storing training sparse depth file paths into: {}'.format(
    TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH, train_sparse_depth_output_paths)

print('Storing training validity map file paths into: {}'.format(
    TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH, train_validity_map_output_paths)

print('Storing training ground truth depth file paths into: {}'.format(
    TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH, train_ground_truth_output_paths)

print('Storing training intrinsics file paths into: {}'.format(
    TRAIN_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_INTRINSICS_OUTPUT_FILEPATH, train_intrinsics_output_paths)


# Write unused file paths
print('Storing unused image file paths into: {}'.format(
    UNUSED_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(
    UNUSED_IMAGE_OUTPUT_FILEPATH, unused_image_output_paths)

print('Storing unused sparse depth file paths into: {}'.format(
    UNUSED_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    UNUSED_SPARSE_DEPTH_OUTPUT_FILEPATH, unused_sparse_depth_output_paths)

print('Storing unused validity map file paths into: {}'.format(
    UNUSED_VALIDITY_MAP_OUTPUT_FILEPATH))
data_utils.write_paths(
    UNUSED_VALIDITY_MAP_OUTPUT_FILEPATH, unused_validity_map_output_paths)

print('Storing unused ground truth file paths into: {}'.format(
    UNUSED_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    UNUSED_GROUND_TRUTH_OUTPUT_FILEPATH, unused_ground_truth_output_paths)

print('Storing unused intrinsics file paths into: {}'.format(
    UNUSED_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    UNUSED_INTRINSICS_OUTPUT_FILEPATH, unused_intrinsics_output_paths)


# Write clean training file paths
print('Removed {} filepaths for clean paths'.format(n_removed))

print('Storing clean training image file paths into: {}'.format(
    TRAIN_IMAGE_CLEAN_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_IMAGE_CLEAN_OUTPUT_FILEPATH, train_image_clean_output_paths)

print('Storing clean training sparse depth file paths into: {}'.format(
    TRAIN_SPARSE_DEPTH_CLEAN_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_SPARSE_DEPTH_CLEAN_OUTPUT_FILEPATH, train_sparse_depth_clean_output_paths)

print('Storing clean training validity map file paths into: {}'.format(
    TRAIN_VALIDITY_MAP_CLEAN_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_VALIDITY_MAP_CLEAN_OUTPUT_FILEPATH, train_validity_map_clean_output_paths)

print('Storing clean training ground truth file paths into: {}'.format(
    TRAIN_GROUND_TRUTH_CLEAN_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_GROUND_TRUTH_CLEAN_OUTPUT_FILEPATH, train_ground_truth_clean_output_paths)

print('Storing clean training intrinsics file paths into: {}'.format(
    TRAIN_INTRINSICS_CLEAN_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_INTRINSICS_CLEAN_OUTPUT_FILEPATH, train_intrinsics_clean_output_paths)


'''
Create validity maps and paths for sparse depth and ground truth for validation and testing
'''
val_image_output_paths = []
val_sparse_depth_output_paths = []
val_validity_map_output_paths = []
val_ground_truth_output_paths = []
val_intrinsics_output_paths = []

test_image_output_paths = []
test_sparse_depth_output_paths = []
test_validity_map_output_paths = []
test_ground_truth_output_paths = []
test_intrinsics_output_paths = []

modes = [
    [
        'validation',
        KITTI_VALIDATION_DIRPATH,
        val_image_output_paths,
        val_sparse_depth_output_paths,
        val_validity_map_output_paths,
        val_ground_truth_output_paths,
        val_intrinsics_output_paths
    ],
    [
        'testing',
        KITTI_TESTING_DIRPATH,
        test_image_output_paths,
        test_sparse_depth_output_paths,
        test_validity_map_output_paths,
        test_ground_truth_output_paths,
        test_intrinsics_output_paths
    ]
]

for mode in modes:
    mode_type, \
        kitti_dirpath, \
        image_output_paths, \
        sparse_depth_output_paths, \
        validity_map_output_paths, \
        ground_truth_output_paths, \
        intrinsics_output_paths = mode

    # Iterate through image, intrinsics, sparse depth and ground truth directories
    for refdir in ['image', 'intrinsics', 'sparse_depth', 'ground_truth']:

        ext = '*.txt' if refdir == 'intrinsics' else '*.png'

        filepaths = sorted(glob.glob(
            os.path.join(kitti_dirpath, refdir, ext)))

        # Iterate filepaths
        for idx in range(len(filepaths)):
            path = filepaths[idx]

            if refdir == 'image':
                image = cv2.imread(path)

                image = np.concatenate([image, image, image], axis=1)
                image_output_path = path \
                    .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH)
                image_output_paths.append(image_output_path)

                image_output_dirpath = os.path.dirname(image_output_path)
                if not os.path.exists(image_output_dirpath):
                    os.makedirs(image_output_dirpath)

                if not args.paths_only:
                    # Write to disk
                    cv2.imwrite(image_output_path, image)

            elif refdir == 'intrinsics':
                intrinsics = np.reshape(np.loadtxt(path), (3, 3))

                intrinsics_output_path = path \
                    .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH) \
                    .replace('.txt', '.npy')

                intrinsics_output_paths.append(intrinsics_output_path)

                if not os.path.exists(os.path.dirname(intrinsics_output_path)):
                    os.makedirs(os.path.dirname(intrinsics_output_path))

                np.save(intrinsics_output_path, intrinsics)

            elif refdir == 'sparse_depth':
                if not args.paths_only:
                    # Load sparse depth and save validity map
                    _, validity_map = data_utils.load_depth_with_validity_map(path)

                # Create validity map output path
                validity_map_output_path = path \
                    .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH) \
                    .replace('sparse_depth', 'validity_map')
                sparse_depth_output_paths.append(path)
                validity_map_output_paths.append(validity_map_output_path)

                validity_map_output_dirpath = os.path.dirname(validity_map_output_path)
                if not os.path.exists(validity_map_output_dirpath):
                    os.makedirs(validity_map_output_dirpath)

                if not args.paths_only:
                    # Write to disk
                    data_utils.save_validity_map(validity_map, validity_map_output_path)

            elif refdir == 'ground_truth':
                ground_truth_output_paths.append(path)

            sys.stdout.write(
                'Processed {}/{} {} {} samples \r'.format(
                    idx + 1, len(filepaths), mode_type, refdir))
            sys.stdout.flush()

        print('Completed generating {} {} {} samples'.format(
            len(filepaths), mode_type, refdir))


# Write validation file paths
print('Storing validation image file paths into: {}'.format(
    VAL_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_IMAGE_OUTPUT_FILEPATH, val_image_output_paths)

print('Storing validation sparse depth file paths into: {}'.format(
    VAL_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_SPARSE_DEPTH_OUTPUT_FILEPATH, val_sparse_depth_output_paths)

print('Storing validation validity map file paths into: {}'.format(
    VAL_VALIDITY_MAP_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_VALIDITY_MAP_OUTPUT_FILEPATH, val_validity_map_output_paths)

print('Storing validation ground truth file paths into: {}'.format(
    VAL_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_GROUND_TRUTH_OUTPUT_FILEPATH, val_ground_truth_output_paths)

print('Storing validation intrinsics file paths into: {}'.format(
    VAL_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_INTRINSICS_OUTPUT_FILEPATH, val_intrinsics_output_paths)


# Write testing file paths
print('Storing testing image file paths into: {}'.format(
    TEST_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(
    TEST_IMAGE_OUTPUT_FILEPATH, test_image_output_paths)

print('Storing testing sparse depth file paths into: {}'.format(
    TEST_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    TEST_SPARSE_DEPTH_OUTPUT_FILEPATH, test_sparse_depth_output_paths)

print('Storing testing validity map file paths into: {}'.format(
    TEST_VALIDITY_MAP_OUTPUT_FILEPATH))
data_utils.write_paths(
    TEST_VALIDITY_MAP_OUTPUT_FILEPATH, test_validity_map_output_paths)

print('Storing testing intrinsics file paths into: {}'.format(
    TEST_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    TEST_INTRINSICS_OUTPUT_FILEPATH, test_intrinsics_output_paths)
