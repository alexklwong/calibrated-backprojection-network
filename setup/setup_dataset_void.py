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
import os, sys, glob, argparse
import multiprocessing as mp
import numpy as np
import cv2
sys.path.insert(0,'src')
# print(sys.path)
import data_utils
from tqdm import tqdm

VOID_ROOT_DIRPATH       = os.path.join('/local1/datasets/Depth_Completion_Datasets/void-dataset/data', 'void_release')
VOID_DATA_1500_DIRPATH  = os.path.join(VOID_ROOT_DIRPATH, 'void_1500')

VOID_OUTPUT_DIRPATH     = os.path.join('/local1/datasets/Depth_Completion_Datasets/void-dataset/data', 'void_kbnet')

VOID_TRAIN_IMAGE_FILENAME         = 'train_image.txt'
VOID_TRAIN_POSE_FILENAME          = 'train_absolute_pose.txt'
VOID_TRAIN_SPARSE_DEPTH_FILENAME  = 'train_sparse_depth.txt'
VOID_TRAIN_VALIDITY_MAP_FILENAME  = 'train_validity_map.txt'
VOID_TRAIN_GROUND_TRUTH_FILENAME  = 'train_ground_truth.txt'
VOID_TRAIN_INTRINSICS_FILENAME    = 'train_intrinsics.txt'
VOID_TEST_IMAGE_FILENAME          = 'test_image.txt'
VOID_TEST_POSE_FILENAME           = 'test_absolute_pose.txt'
VOID_TEST_SPARSE_DEPTH_FILENAME   = 'test_sparse_depth.txt'
VOID_TEST_VALIDITY_MAP_FILENAME   = 'test_validity_map.txt'
VOID_TEST_GROUND_TRUTH_FILENAME   = 'test_ground_truth.txt'
VOID_TEST_INTRINSICS_FILENAME     = 'test_intrinsics.txt'

TRAIN_REFS_DIRPATH      = os.path.join('training', 'void')
TEST_REFS_DIRPATH       = os.path.join('testing', 'void')

# VOID training set 1500 density
VOID_TRAIN_IMAGE_1500_FILEPATH          = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_image_1500.txt')
VOID_TRAIN_POSE_1500_FILEPATH           = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_pose_1500.txt')
VOID_TRAIN_SPARSE_DEPTH_1500_FILEPATH   = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_sparse_depth_1500.txt')
VOID_TRAIN_VALIDITY_MAP_1500_FILEPATH   = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_validity_map_1500.txt')
VOID_TRAIN_GROUND_TRUTH_1500_FILEPATH   = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_ground_truth_1500.txt')
VOID_TRAIN_INTRINSICS_1500_FILEPATH     = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_intrinsics_1500.txt')
# VOID testing set 1500 density
VOID_TEST_IMAGE_1500_FILEPATH           = os.path.join(TEST_REFS_DIRPATH, 'void_test_image_1500.txt')
VOID_TEST_POSE_1500_FILEPATH            = os.path.join(TEST_REFS_DIRPATH, 'void_test_pose_1500.txt')
VOID_TEST_SPARSE_DEPTH_1500_FILEPATH    = os.path.join(TEST_REFS_DIRPATH, 'void_test_sparse_depth_1500.txt')
VOID_TEST_VALIDITY_MAP_1500_FILEPATH    = os.path.join(TEST_REFS_DIRPATH, 'void_test_validity_map_1500.txt')
VOID_TEST_GROUND_TRUTH_1500_FILEPATH    = os.path.join(TEST_REFS_DIRPATH, 'void_test_ground_truth_1500.txt')
VOID_TEST_INTRINSICS_1500_FILEPATH      = os.path.join(TEST_REFS_DIRPATH, 'void_test_intrinsics_1500.txt')
# VOID unused testing set 1500 density
VOID_UNUSED_IMAGE_1500_FILEPATH         = os.path.join(TEST_REFS_DIRPATH, 'void_unused_image_1500.txt')
VOID_UNUSED_POSE_1500_FILEPATH          = os.path.join(TEST_REFS_DIRPATH, 'void_unused_pose_1500.txt')
VOID_UNUSED_SPARSE_DEPTH_1500_FILEPATH  = os.path.join(TEST_REFS_DIRPATH, 'void_unused_sparse_depth_1500.txt')
VOID_UNUSED_VALIDITY_MAP_1500_FILEPATH  = os.path.join(TEST_REFS_DIRPATH, 'void_unused_validity_map_1500.txt')
VOID_UNUSED_GROUND_TRUTH_1500_FILEPATH  = os.path.join(TEST_REFS_DIRPATH, 'void_unused_ground_truth_1500.txt')
VOID_UNUSED_INTRINSICS_1500_FILEPATH    = os.path.join(TEST_REFS_DIRPATH, 'void_unused_intrinsics_1500.txt')


def process_frame(inputs):
    '''
    Processes a single frame

    Arg(s):
        inputs : tuple
            image path at time t=0,
            image path at time t=1,
            image path at time t=-1,
            pose path at time t=0
            pose path at time t=1
            pose path at time t=-1
            sparse depth path at time t=0,
            validity map path at time t=0,
            ground truth path at time t=0,
            boolean flag if set then create paths only
    Returns:
        str : image reference directory path
        str : output concatenated image path at time t=0
        str : output sparse depth path at time t=0
        str : output validity map path at time t=0
        str : output ground truth path at time t=0
    '''

    image_path1, \
        image_path0, \
        image_path2, \
        pose_path1, \
        pose_path0, \
        pose_path2, \
        sparse_depth_path, \
        validity_map_path, \
        ground_truth_path, \
        paths_only = inputs

    if not paths_only:
        # Create image composite of triplets
        image1 = cv2.imread(image_path1)
        image0 = cv2.imread(image_path0)
        image2 = cv2.imread(image_path2)
        imagec = np.concatenate([image1, image0, image2], axis=1)
        # Store poses of image triplet
        pose1 = np.loadtxt(pose_path1)
        pose0 = np.loadtxt(pose_path0)
        pose2 = np.loadtxt(pose_path2)
        posec = np.stack((pose1, pose0, pose2))
        posec = posec.reshape(posec.shape[0],-1)
        # Get validity map
        sparse_depth, validity_map = data_utils.load_depth_with_validity_map(sparse_depth_path)

    # print("IMAGE PATH: ", *image_path0.split(os.sep)[8:])
    image_refpath = os.path.join(*image_path0.split(os.sep)[8:])
    pose_refpath  = os.path.join(*pose_path0.split(os.sep)[8:])
    # Set output paths
    image_outpath = os.path.join(VOID_OUTPUT_DIRPATH, image_refpath)
    pose_outpath  = os.path.join(VOID_OUTPUT_DIRPATH, pose_refpath)    
    sparse_depth_outpath = sparse_depth_path
    validity_map_outpath = validity_map_path
    ground_truth_outpath = ground_truth_path

    # Verify that all filenames match
    image_out_dirpath, image_filename = os.path.split(image_outpath)
    pose_out_dirpath, pose_filename = os.path.split(pose_outpath)
    sparse_depth_filename = os.path.basename(sparse_depth_outpath)
    validity_map_filename = os.path.basename(validity_map_outpath)
    ground_truth_filename = os.path.basename(ground_truth_outpath)

    assert os.path.splitext(image_filename)[0] == os.path.splitext(pose_filename)[0]
    assert image_filename == sparse_depth_filename
    assert image_filename == validity_map_filename
    assert image_filename == ground_truth_filename

    if not paths_only:
        cv2.imwrite(image_outpath, imagec)
        np.savetxt(pose_outpath, posec)
    return (image_refpath,
            image_outpath,
            pose_refpath,
            pose_outpath,
            sparse_depth_outpath,
            validity_map_outpath,
            ground_truth_outpath)


parser = argparse.ArgumentParser()

parser.add_argument('--paths_only', action='store_true')

args = parser.parse_args()


data_dirpaths = [
    VOID_DATA_1500_DIRPATH
]

train_output_filepaths = [
    [
        VOID_TRAIN_IMAGE_1500_FILEPATH,
        VOID_TRAIN_POSE_1500_FILEPATH,
        VOID_TRAIN_SPARSE_DEPTH_1500_FILEPATH,
        VOID_TRAIN_VALIDITY_MAP_1500_FILEPATH,
        VOID_TRAIN_GROUND_TRUTH_1500_FILEPATH,
        VOID_TRAIN_INTRINSICS_1500_FILEPATH
    ]
]
test_output_filepaths = [
    [
        VOID_TEST_IMAGE_1500_FILEPATH,
        VOID_TEST_POSE_1500_FILEPATH,
        VOID_TEST_SPARSE_DEPTH_1500_FILEPATH,
        VOID_TEST_VALIDITY_MAP_1500_FILEPATH,
        VOID_TEST_GROUND_TRUTH_1500_FILEPATH,
        VOID_TEST_INTRINSICS_1500_FILEPATH
    ]
]
unused_output_filepaths = [
    [
        VOID_UNUSED_IMAGE_1500_FILEPATH,
        VOID_UNUSED_POSE_1500_FILEPATH,
        VOID_UNUSED_SPARSE_DEPTH_1500_FILEPATH,
        VOID_UNUSED_VALIDITY_MAP_1500_FILEPATH,
        VOID_UNUSED_GROUND_TRUTH_1500_FILEPATH,
        VOID_UNUSED_INTRINSICS_1500_FILEPATH
    ]
]

for dirpath in tqdm([TRAIN_REFS_DIRPATH, TEST_REFS_DIRPATH]):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

data_filepaths = \
    zip(data_dirpaths, train_output_filepaths, test_output_filepaths, unused_output_filepaths)

for data_dirpath, train_filepaths, test_filepaths, unused_filepaths in tqdm(data_filepaths):
    # Training set
    train_image_filepath = os.path.join(data_dirpath, VOID_TRAIN_IMAGE_FILENAME)
    train_pose_filepath  = os.path.join(data_dirpath, VOID_TRAIN_POSE_FILENAME)
    train_sparse_depth_filepath = os.path.join(data_dirpath, VOID_TRAIN_SPARSE_DEPTH_FILENAME)
    train_validity_map_filepath = os.path.join(data_dirpath, VOID_TRAIN_VALIDITY_MAP_FILENAME)
    train_ground_truth_filepath = os.path.join(data_dirpath, VOID_TRAIN_GROUND_TRUTH_FILENAME)
    train_intrinsics_filepath = os.path.join(data_dirpath, VOID_TRAIN_INTRINSICS_FILENAME)

    # Read training paths
    train_image_paths = data_utils.read_paths(train_image_filepath)
    # combined_trainImage = '\t'.join(train_image_paths)
    train_pose_paths = data_utils.read_paths(train_pose_filepath)
    # combined_trainPose = '\t'.join(train_pose_paths)
    train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_filepath)
    # combined_trainSparse = '\t'.join(train_sparse_depth_paths)
    train_validity_map_paths = data_utils.read_paths(train_validity_map_filepath)
    # combined_trainVal = '\t'.join(train_validity_map_paths)
    train_ground_truth_paths = data_utils.read_paths(train_ground_truth_filepath)
    # combined_trainGT = '\t'.join(train_ground_truth_paths)
    train_intrinsics_paths = data_utils.read_paths(train_intrinsics_filepath)
    # combined_trainIn = '\t'.join(train_intrinsics_paths)

    assert len(train_image_paths) == len(train_pose_paths)
    assert len(train_image_paths) == len(train_sparse_depth_paths)
    assert len(train_image_paths) == len(train_validity_map_paths)
    assert len(train_image_paths) == len(train_ground_truth_paths)
    assert len(train_image_paths) == len(train_intrinsics_paths)

    # Testing set
    test_image_filepath = os.path.join(data_dirpath, VOID_TEST_IMAGE_FILENAME)
    test_pose_filepath  = os.path.join(data_dirpath, VOID_TEST_POSE_FILENAME)
    test_sparse_depth_filepath = os.path.join(data_dirpath, VOID_TEST_SPARSE_DEPTH_FILENAME)
    test_validity_map_filepath = os.path.join(data_dirpath, VOID_TEST_VALIDITY_MAP_FILENAME)
    test_ground_truth_filepath = os.path.join(data_dirpath, VOID_TEST_GROUND_TRUTH_FILENAME)
    test_intrinsics_filepath = os.path.join(data_dirpath, VOID_TEST_INTRINSICS_FILENAME)

    # Read testing paths
    test_image_paths = data_utils.read_paths(test_image_filepath)
    test_pose_paths = data_utils.read_paths(test_pose_filepath)
    test_sparse_depth_paths = data_utils.read_paths(test_sparse_depth_filepath)
    test_validity_map_paths = data_utils.read_paths(test_validity_map_filepath)
    test_ground_truth_paths = data_utils.read_paths(test_ground_truth_filepath)
    test_intrinsics_paths = data_utils.read_paths(test_intrinsics_filepath)

    assert len(test_image_paths) == len(test_pose_paths)
    assert len(test_image_paths) == len(test_sparse_depth_paths)
    assert len(test_image_paths) == len(test_validity_map_paths)
    assert len(test_image_paths) == len(test_ground_truth_paths)
    assert len(test_image_paths) == len(test_intrinsics_paths)

    # Get test set directories
    test_seq_dirpaths = set(
        [test_image_paths[idx].split(os.sep)[-3] for idx in range(len(test_image_paths))])

    # Initialize placeholders for training output paths
    train_image_outpaths = []
    train_pose_outpaths = []
    train_sparse_depth_outpaths = []
    train_validity_map_outpaths = []
    train_ground_truth_outpaths = []
    train_intrinsics_outpaths = []

    # Initialize placeholders for testing output paths
    test_image_outpaths = []
    test_pose_outpaths = []
    test_sparse_depth_outpaths = []
    test_validity_map_outpaths = []
    test_ground_truth_outpaths = []
    test_intrinsics_outpaths = []

    # Initialize placeholders for unused testing output paths
    unused_image_outpaths = []
    unused_pose_outpaths = []
    unused_sparse_depth_outpaths = []
    unused_validity_map_outpaths = []
    unused_ground_truth_outpaths = []
    unused_intrinsics_outpaths = []

    # For each dataset density, grab the sequences
    seq_dirpaths = glob.glob(os.path.join(data_dirpath, 'data', '*'))
    n_sample = 0

    for seq_dirpath in tqdm(seq_dirpaths):
        # For each sequence, grab the images, sparse depths and valid maps
        image_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'image', '*.png')))
        pose_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'absolute_pose', '*.txt')))
        sparse_depth_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'sparse_depth', '*.png')))
        validity_map_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'validity_map', '*.png')))
        ground_truth_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'ground_truth', '*.png')))
        intrinsics_path = os.path.join(seq_dirpath, 'K.txt')

        assert len(image_paths) == len(pose_paths)
        assert len(image_paths) == len(sparse_depth_paths)
        assert len(image_paths) == len(validity_map_paths)

        # Load intrinsics
        kin = np.loadtxt(intrinsics_path)

        intrinsics_refpath = \
            os.path.join(*intrinsics_path.split(os.sep)[8:])
        intrinsics_outpath = \
            os.path.join(VOID_OUTPUT_DIRPATH, intrinsics_refpath[:-3]+'npy')
        image_out_dirpath = \
            os.path.join(os.path.dirname(intrinsics_outpath), 'image')
        pose_out_dirpath = \
            os.path.join(os.path.dirname(intrinsics_outpath), 'absolute_pose')

        # print(image_out_dirpath)

        if not os.path.exists(image_out_dirpath):
            os.makedirs(image_out_dirpath)
        if not os.path.exists(pose_out_dirpath):
            os.makedirs(pose_out_dirpath)

        # Save intrinsics
        np.save(intrinsics_outpath, kin)

        if seq_dirpath.split(os.sep)[-1] in test_seq_dirpaths:
            start_idx = 0
            offset_idx = 0
        else:
            # Skip first stationary 30 frames (1 second) and skip every 10
            start_idx = 30
            offset_idx = 10

        pool_input = []
        for idx in tqdm(range(start_idx, len(image_paths)-offset_idx-start_idx)):
            pool_input.append((
                image_paths[idx-offset_idx],
                image_paths[idx],
                image_paths[idx+offset_idx],
                pose_paths[idx-offset_idx],
                pose_paths[idx],
                pose_paths[idx+offset_idx],
                sparse_depth_paths[idx],
                validity_map_paths[idx],
                ground_truth_paths[idx],
                args.paths_only))

        with mp.Pool() as pool:
            pool_results = pool.map(process_frame, pool_input)

            for result in tqdm(pool_results):
                image_refpath, \
                    image_outpath, \
                    pose_refpath, \
                    pose_outpath, \
                    sparse_depth_outpath, \
                    validity_map_outpath, \
                    ground_truth_outpath = result

                # Split into training, testing and unused testing sets
                if 'void_1500/'+image_refpath in train_image_paths:
                    train_image_outpaths.append(image_outpath)
                    train_pose_outpaths.append(pose_outpath)
                    train_sparse_depth_outpaths.append(sparse_depth_outpath)
                    train_validity_map_outpaths.append(validity_map_outpath)
                    train_ground_truth_outpaths.append(ground_truth_outpath)
                    train_intrinsics_outpaths.append(intrinsics_outpath)
                elif 'void_1500/'+image_refpath in test_image_paths:
                    test_image_outpaths.append(image_outpath)
                    test_pose_outpaths.append(pose_outpath)
                    test_sparse_depth_outpaths.append(sparse_depth_outpath)
                    test_validity_map_outpaths.append(validity_map_outpath)
                    test_ground_truth_outpaths.append(ground_truth_outpath)
                    test_intrinsics_outpaths.append(intrinsics_outpath)
                else:
                    unused_image_outpaths.append(image_outpath)
                    unused_pose_outpaths.append(pose_outpath)
                    unused_sparse_depth_outpaths.append(sparse_depth_outpath)
                    unused_validity_map_outpaths.append(validity_map_outpath)
                    unused_ground_truth_outpaths.append(ground_truth_outpath)
                    unused_intrinsics_outpaths.append(intrinsics_outpath)

        n_sample = n_sample + len(pool_input)

        print('Completed processing {} examples for sequence={}'.format(
            len(pool_input), seq_dirpath))

    print('Completed processing {} examples for density={}'.format(n_sample, data_dirpath))

    void_train_image_filepath, \
        void_train_pose_filepath, \
        void_train_sparse_depth_filepath, \
        void_train_validity_map_filepath, \
        void_train_ground_truth_filepath, \
        void_train_intrinsics_filepath = train_filepaths

    print('Storing {} training image file paths into: {}'.format(
        len(train_image_outpaths), void_train_image_filepath))
    data_utils.write_paths(
        void_train_image_filepath, train_image_outpaths)

    print('Storing {} training pose file paths into: {}'.format(
        len(train_pose_outpaths), void_train_pose_filepath))
    data_utils.write_paths(
        void_train_pose_filepath, train_pose_outpaths)

    print('Storing {} training sparse depth file paths into: {}'.format(
        len(train_sparse_depth_outpaths), void_train_sparse_depth_filepath))
    data_utils.write_paths(
        void_train_sparse_depth_filepath, train_sparse_depth_outpaths)

    print('Storing {} training validity map file paths into: {}'.format(
        len(train_validity_map_outpaths), void_train_validity_map_filepath))
    data_utils.write_paths(
        void_train_validity_map_filepath, train_validity_map_outpaths)

    print('Storing {} training groundtruth depth file paths into: {}'.format(
        len(train_ground_truth_outpaths), void_train_ground_truth_filepath))
    data_utils.write_paths(
        void_train_ground_truth_filepath, train_ground_truth_outpaths)

    print('Storing {} training camera intrinsics file paths into: {}'.format(
        len(train_intrinsics_outpaths), void_train_intrinsics_filepath))
    data_utils.write_paths(
        void_train_intrinsics_filepath, train_intrinsics_outpaths)

    void_test_image_filepath, \
        void_test_pose_filepath, \
        void_test_sparse_depth_filepath, \
        void_test_validity_map_filepath, \
        void_test_ground_truth_filepath, \
        void_test_intrinsics_filepath = test_filepaths

    print('Storing {} testing image file paths into: {}'.format(
        len(test_image_outpaths), void_test_image_filepath))
    data_utils.write_paths(
        void_test_image_filepath, test_image_outpaths)

    print('Storing {} testing pose file paths into: {}'.format(
        len(test_pose_outpaths), void_test_pose_filepath))
    data_utils.write_paths(
        void_test_pose_filepath, test_pose_outpaths)

    print('Storing {} testing sparse depth file paths into: {}'.format(
        len(test_sparse_depth_outpaths), void_test_sparse_depth_filepath))
    data_utils.write_paths(
        void_test_sparse_depth_filepath, test_sparse_depth_outpaths)

    print('Storing {} testing validity map file paths into: {}'.format(
        len(test_validity_map_outpaths), void_test_validity_map_filepath))
    data_utils.write_paths(
        void_test_validity_map_filepath, test_validity_map_outpaths)

    print('Storing {} testing groundtruth depth file paths into: {}'.format(
        len(test_ground_truth_outpaths), void_test_ground_truth_filepath))
    data_utils.write_paths(
        void_test_ground_truth_filepath, test_ground_truth_outpaths)

    print('Storing {} testing camera intrinsics file paths into: {}'.format(
        len(test_intrinsics_outpaths), void_test_intrinsics_filepath))
    data_utils.write_paths(
        void_test_intrinsics_filepath, test_intrinsics_outpaths)

    void_unused_image_filepath, \
        void_unused_pose_filepath, \
        void_unused_sparse_depth_filepath, \
        void_unused_validity_map_filepath, \
        void_unused_ground_truth_filepath, \
        void_unused_intrinsics_filepath = unused_filepaths

    print('Storing {} unused testing image file paths into: {}'.format(
        len(unused_image_outpaths), void_unused_image_filepath))
    data_utils.write_paths(
        void_unused_image_filepath, unused_image_outpaths)

    print('Storing {} unused testing pose file paths into: {}'.format(
        len(unused_pose_outpaths), void_unused_pose_filepath))
    data_utils.write_paths(
        void_unused_pose_filepath, unused_pose_outpaths)

    print('Storing {} unused testing sparse depth file paths into: {}'.format(
        len(unused_sparse_depth_outpaths), void_unused_sparse_depth_filepath))
    data_utils.write_paths(
        void_unused_sparse_depth_filepath, unused_sparse_depth_outpaths)

    print('Storing {} unused testing validity map file paths into: {}'.format(
        len(unused_validity_map_outpaths), void_unused_validity_map_filepath))
    data_utils.write_paths(
        void_unused_validity_map_filepath, unused_validity_map_outpaths)

    print('Storing {} unused testing groundtruth depth file paths into: {}'.format(
        len(unused_ground_truth_outpaths), void_unused_ground_truth_filepath))
    data_utils.write_paths(
        void_unused_ground_truth_filepath, unused_ground_truth_outpaths)

    print('Storing {} unused testing camera intrinsics file paths into: {}'.format(
        len(unused_intrinsics_outpaths), void_unused_intrinsics_filepath))
    data_utils.write_paths(
        void_unused_intrinsics_filepath, unused_intrinsics_outpaths)