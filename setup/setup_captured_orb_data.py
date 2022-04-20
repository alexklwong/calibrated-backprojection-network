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
from pathlib import Path as path 
import data_utils
from tqdm import tqdm
from natsort import natsorted

paths_only = False
skip = 5

def create_imagesPaths_files(DATA_DIRPATH, train_fileNames, test_fileNames):
    train_image_fileName, train_pose_fileName, train_sparse_depth_fileName, train_validity_map_fileName, train_ground_truth_fileName, train_intrinsics_fileName  = train_fileNames   
    test_image_fileName, test_pose_fileName, test_sparse_depth_fileName, test_validity_map_fileName, test_ground_truth_fileName, test_intrinsics_fileName  = test_fileNames   
    
    train_image_filepath = os.path.join(DATA_DIRPATH, train_image_fileName)
    train_pose_filepath = os.path.join(DATA_DIRPATH, train_pose_fileName)
    train_sparse_depth_filepath = os.path.join(DATA_DIRPATH, train_sparse_depth_fileName)
    train_validity_map_filepath = os.path.join(DATA_DIRPATH, train_validity_map_fileName)
    train_ground_truth_filepath = os.path.join(DATA_DIRPATH, train_ground_truth_fileName)
    train_intrinsics_filepath = os.path.join(DATA_DIRPATH, train_intrinsics_fileName)

    test_image_filepath = os.path.join(DATA_DIRPATH, test_image_fileName)
    test_pose_filepath = os.path.join(DATA_DIRPATH, test_pose_fileName)
    test_sparse_depth_filepath = os.path.join(DATA_DIRPATH, test_sparse_depth_fileName)
    test_validity_map_filepath = os.path.join(DATA_DIRPATH, test_validity_map_fileName)
    test_ground_truth_filepath = os.path.join(DATA_DIRPATH, test_ground_truth_fileName)
    test_intrinsics_filepath = os.path.join(DATA_DIRPATH, test_intrinsics_fileName)


    train_imageF = open(train_image_filepath, "w")
    train_poseF = open(train_pose_filepath, "w")
    train_sparseF = open(train_sparse_depth_filepath, "w")
    train_valF = open(train_validity_map_filepath, "w")
    train_gtF = open(train_ground_truth_filepath, "w")
    train_KF = open(train_intrinsics_filepath, "w")
    
    test_imageF = open(test_image_filepath, "w")
    test_poseF = open(test_pose_filepath, "w")
    test_sparseF = open(test_sparse_depth_filepath, "w")
    test_valF = open(test_validity_map_filepath, "w")
    test_gtF = open(test_ground_truth_filepath, "w")
    test_KF = open(test_intrinsics_filepath, "w")

    images_folder = os.path.join(DATA_DIRPATH, "images")
    poses_folder = os.path.join(DATA_DIRPATH, "poses")
    depths_folder = os.path.join(DATA_DIRPATH, "depths")
    val_folder = os.path.join(DATA_DIRPATH, "validity_maps")
    gt_folder = os.path.join(DATA_DIRPATH, "raw_depths")

    numImages = len([i for i in os.listdir(images_folder)])
    train_imagePaths = []
    train_posePaths = []
    train_sparsePaths = []
    train_valPaths = []
    train_GTPaths = []
    train_intrinsicPaths = []
    
    test_imagePaths = []
    test_posePaths = []
    test_sparsePaths = []
    test_valPaths = []
    test_GTPaths = []
    test_intrinsicPaths = []

    
    training_indices = np.random.choice(numImages, int(0.9*numImages), replace=False)
    for i in range(numImages):
        if i< skip:
            continue
        if i in training_indices:
            train_imagePaths.append( os.path.join(images_folder, "keyframe"+str(i)+".png\n")  )
            train_posePaths.append( os.path.join(poses_folder, "pose"+str(i)+".txt\n")  )
            train_sparsePaths.append( os.path.join(depths_folder, "depth"+str(i)+".yml\n")  )
            train_valPaths.append( os.path.join(val_folder, "validity_map"+str(i)+".png\n")  )
            train_GTPaths.append( os.path.join(gt_folder, "rawDepth"+str(i)+".yml\n")  )
            train_intrinsicPaths.append( os.path.join(DATA_DIRPATH, "intrinsics.yml\n")  )
        else:
            test_imagePaths.append(os.path.join(images_folder, "keyframe"+str(i)+".png\n"))
            test_posePaths.append( os.path.join(poses_folder, "pose"+str(i)+".txt\n")  )
            test_sparsePaths.append( os.path.join(depths_folder, "depth"+str(i)+".yml\n")  )
            test_valPaths.append( os.path.join(val_folder, "validity_map"+str(i)+".png\n")  )
            test_GTPaths.append( os.path.join(gt_folder, "rawDepth"+str(i)+".yml\n")  )
            test_intrinsicPaths.append( os.path.join(DATA_DIRPATH, "intrinsics.yml\n")  )

    train_imageF.writelines(train_imagePaths)
    train_poseF.writelines(train_posePaths)
    train_sparseF.writelines(train_sparsePaths)
    train_valF.writelines(train_valPaths) 
    train_gtF.writelines(train_GTPaths)
    train_KF.writelines(train_intrinsicPaths)
    
    test_imageF.writelines(test_imagePaths)
    test_poseF.writelines(test_posePaths)
    test_sparseF.writelines(test_sparsePaths)
    test_valF.writelines(test_valPaths) 
    test_gtF.writelines(test_GTPaths)
    test_KF.writelines(test_intrinsicPaths)

    train_imageF.close()
    train_poseF.close()
    train_sparseF.close()
    train_valF.close()
    train_gtF.close()
    train_KF.close()
    test_imageF.close()
    test_poseF.close()
    test_sparseF.close()
    test_valF.close()
    test_gtF.close()
    test_KF.close()

DATA_DIRPATH  = sys.argv[1]
OUTPUT_DIRPATH     = sys.argv[2] 

TRAIN_IMAGE_FILENAME         = 'train_image.txt'
TRAIN_POSE_FILENAME          = 'train_pose.txt'
TRAIN_SPARSE_DEPTH_FILENAME  = 'train_sparse_depth.txt'
TRAIN_VALIDITY_MAP_FILENAME  = 'train_validity_map.txt'
TRAIN_GROUND_TRUTH_FILENAME  = 'train_ground_truth.txt'
TRAIN_INTRINSICS_FILENAME    = 'train_intrinsics.txt'
TEST_IMAGE_FILENAME          = 'test_image.txt'
TEST_POSE_FILENAME           = 'test_pose.txt'
TEST_SPARSE_DEPTH_FILENAME   = 'test_sparse_depth.txt'
TEST_VALIDITY_MAP_FILENAME   = 'test_validity_map.txt'
TEST_GROUND_TRUTH_FILENAME   = 'test_ground_truth.txt'
TEST_INTRINSICS_FILENAME     = 'test_intrinsics.txt'


TRAIN_REFS_DIRPATH      = os.path.join('training', 'orb')
TEST_REFS_DIRPATH       = os.path.join('testing', 'orb')


# VOID training set 1500 density
TRAIN_IMAGE_FILEPATH          = os.path.join(TRAIN_REFS_DIRPATH, 'train_image.txt')
TRAIN_POSE_FILEPATH           = os.path.join(TRAIN_REFS_DIRPATH, 'train_pose.txt')
TRAIN_SPARSE_DEPTH_FILEPATH   = os.path.join(TRAIN_REFS_DIRPATH, 'train_sparse_depth.txt')
TRAIN_VALIDITY_MAP_FILEPATH   = os.path.join(TRAIN_REFS_DIRPATH, 'train_validity_map.txt')
TRAIN_GROUND_TRUTH_FILEPATH   = os.path.join(TRAIN_REFS_DIRPATH, 'train_ground_truth.txt')
TRAIN_INTRINSICS_FILEPATH    = os.path.join(TRAIN_REFS_DIRPATH, 'train_intrinsics.txt')
# VOID testing set 1500 density
TEST_IMAGE_FILEPATH           = os.path.join(TEST_REFS_DIRPATH, 'test_image.txt')
TEST_POSE_FILEPATH            = os.path.join(TEST_REFS_DIRPATH, 'test_pose.txt')
TEST_SPARSE_DEPTH_FILEPATH    = os.path.join(TEST_REFS_DIRPATH, 'test_sparse_depth.txt')
TEST_VALIDITY_MAP_FILEPATH    = os.path.join(TEST_REFS_DIRPATH, 'test_validity_map.txt')
TEST_GROUND_TRUTH_FILEPATH    = os.path.join(TEST_REFS_DIRPATH, 'test_ground_truth.txt')
TEST_INTRINSICS_FILEPATH      = os.path.join(TEST_REFS_DIRPATH, 'test_intrinsics.txt')

training_names = [TRAIN_IMAGE_FILENAME, TRAIN_POSE_FILENAME, TRAIN_SPARSE_DEPTH_FILENAME, TRAIN_VALIDITY_MAP_FILENAME, TRAIN_GROUND_TRUTH_FILENAME, TRAIN_INTRINSICS_FILENAME]
testing_names = [TEST_IMAGE_FILENAME, TEST_POSE_FILENAME, TEST_SPARSE_DEPTH_FILENAME, TEST_VALIDITY_MAP_FILENAME, TEST_GROUND_TRUTH_FILENAME, TEST_INTRINSICS_FILENAME]
create_imagesPaths_files(DATA_DIRPATH,  training_names,testing_names)

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
        
        if path(pose_path0).suffix.lower()==".txt": 
            pose1 = np.loadtxt(pose_path1)
            pose0 = np.loadtxt(pose_path0)
            pose2 = np.loadtxt(pose_path2)
            posec = np.stack((pose1, pose0, pose2))
            posec = posec.reshape(posec.shape[0],-1)
        elif path(pose_path0).suffix.lower()==".yml":
            s = cv2.FileStorage()
            _ = s.open(pose_path1, cv2.FileStorage_READ)
            assert _ == True
            pnode = s.getNode('camera_pose')
            pose1 = pnode.mat()
            _ = s.open(pose_path0, cv2.FileStorage_READ)
            assert _ == True
            pnode = s.getNode('camera_pose')
            pose0 = pnode.mat()
            _ = s.open(pose_path2, cv2.FileStorage_READ)
            assert _ == True
            pnode = s.getNode('camera_pose')
            pose2 = pnode.mat()
            posec = np.stack((pose1, pose0, pose2))
            posec = posec.reshape(posec.shape[0],-1)

        # Get validity map
        sparse_depth, validity_map = data_utils.load_depth_with_validity_map(sparse_depth_path, file_format='yml')

    # image_refpath = os.path.join(*image_path0.split(os.sep)[8:])
    # pose_refpath  = os.path.join(*pose_path0.split(os.sep)[8:])
    imageName = path(image_path0).name
    poseName  = path(pose_path0).name.replace("yml", "txt")

    # Set output paths
    image_outpath = path.joinpath(path(OUTPUT_DIRPATH), "images", imageName)
    pose_outpath  = path.joinpath(path(OUTPUT_DIRPATH), "poses", poseName)      
    sparse_depth_outpath = sparse_depth_path
    validity_map_outpath = validity_map_path
    ground_truth_outpath = ground_truth_path

    # Verify that all filenames match
    image_filename = imageName
    pose_filename  = poseName
    sparse_depth_filename = path(sparse_depth_outpath).name
    validity_map_filename = path(validity_map_outpath).name
    ground_truth_filename = path(ground_truth_outpath).name
    assert "".join(filter(str.isdigit, path(image_filename).name.replace(path(image_filename).suffix, " ")) ) == "".join(filter(str.isdigit, path(pose_filename).name.replace(path(pose_filename).suffix, " ")) )
    assert "".join(filter(str.isdigit, image_filename) ) == "".join(filter(str.isdigit, sparse_depth_filename) )
    assert "".join(filter(str.isdigit, image_filename) ) == "".join(filter(str.isdigit, validity_map_filename) )
    assert "".join(filter(str.isdigit, image_filename) ) == "".join(filter(str.isdigit, ground_truth_filename) )

    if not paths_only:
        cv2.imwrite(str(image_outpath), imagec)
        np.savetxt(str(pose_outpath), posec)
    return (imageName,
            image_outpath,
            poseName,
            pose_outpath,
            sparse_depth_outpath,
            validity_map_outpath,
            ground_truth_outpath)


# parser = argparse.ArgumentParser()
# parser.add_argument('--paths_only', action='store_true')

# args = parser.parse_args()

data_dirpaths = [
    DATA_DIRPATH
]

train_output_filepaths = [
    [
        TRAIN_IMAGE_FILEPATH,
        TRAIN_POSE_FILEPATH,
        TRAIN_SPARSE_DEPTH_FILEPATH,
        TRAIN_VALIDITY_MAP_FILEPATH,
        TRAIN_GROUND_TRUTH_FILEPATH,
        TRAIN_INTRINSICS_FILEPATH
    ]
]
test_output_filepaths = [
    [
        TEST_IMAGE_FILEPATH,
        TEST_POSE_FILEPATH,
        TEST_SPARSE_DEPTH_FILEPATH,
        TEST_VALIDITY_MAP_FILEPATH,
        TEST_GROUND_TRUTH_FILEPATH,
        TEST_INTRINSICS_FILEPATH
    ]
]

for dirpath in tqdm([TRAIN_REFS_DIRPATH, TEST_REFS_DIRPATH]):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

data_filepaths = \
    zip(data_dirpaths, train_output_filepaths, test_output_filepaths)

for data_dirpath, train_filepaths, test_filepaths in tqdm(data_filepaths):
    # Training set
    train_image_filepath = os.path.join(data_dirpath, TRAIN_IMAGE_FILENAME)
    train_pose_filepath  = os.path.join(data_dirpath, TRAIN_POSE_FILENAME)
    train_sparse_depth_filepath = os.path.join(data_dirpath, TRAIN_SPARSE_DEPTH_FILENAME)
    train_validity_map_filepath = os.path.join(data_dirpath, TRAIN_VALIDITY_MAP_FILENAME)
    train_ground_truth_filepath = os.path.join(data_dirpath, TRAIN_GROUND_TRUTH_FILENAME)
    train_intrinsics_filepath = os.path.join(data_dirpath, TRAIN_INTRINSICS_FILENAME)

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
    test_image_filepath = os.path.join(data_dirpath, TEST_IMAGE_FILENAME)
    test_pose_filepath  = os.path.join(data_dirpath, TEST_POSE_FILENAME)
    test_sparse_depth_filepath = os.path.join(data_dirpath, TEST_SPARSE_DEPTH_FILENAME)
    test_validity_map_filepath = os.path.join(data_dirpath, TEST_VALIDITY_MAP_FILENAME)
    test_ground_truth_filepath = os.path.join(data_dirpath, TEST_GROUND_TRUTH_FILENAME)
    test_intrinsics_filepath = os.path.join(data_dirpath, TEST_INTRINSICS_FILENAME)

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

    # For each dataset density, grab the sequences
    seq_dirpaths = [DATA_DIRPATH]#glob.glob(os.path.join(data_dirpath, 'data', '*'))
    n_sample = 0

    for seq_dirpath in tqdm(seq_dirpaths):
        # For each sequence, grab the images, sparse depths and valid maps
        image_paths = \
            natsorted(glob.glob(os.path.join(seq_dirpath, 'images', '*.png')))
        pose_paths = \
            natsorted(glob.glob(os.path.join(seq_dirpath, 'poses', '*.yml')))
        sparse_depth_paths = \
            natsorted(glob.glob(os.path.join(seq_dirpath, 'depths', '*.yml')))
        validity_map_paths = \
            natsorted(glob.glob(os.path.join(seq_dirpath, 'validity_maps', '*.png')))
        ground_truth_paths = \
            natsorted(glob.glob(os.path.join(seq_dirpath, 'raw_depths', '*.yml')))
        intrinsics_path = os.path.join(seq_dirpath, 'intrinsics.yml')

        assert len(image_paths) == len(pose_paths)
        assert len(image_paths) == len(sparse_depth_paths)
        assert len(image_paths) == len(validity_map_paths)

        # Load intrinsics
        if path(intrinsics_path).suffix.lower()==".txt":
            kin = np.loadtxt(intrinsics_path)
        elif path(intrinsics_path).suffix.lower()==".yml":
            s = cv2.FileStorage()
            _ = s.open(intrinsics_path, cv2.FileStorage_READ)
            Knode = s.getNode('camera_intrinsics')
            kin   = Knode.mat()
            if kin.shape[1]==4:
                kin = kin[:,:-1]
        intrinsics_path_obj = path(intrinsics_path)
        intrinsics_file = intrinsics_path_obj.name
        intrinsics_refpath = \
            intrinsics_file
        intrinsics_outpath = \
            os.path.join(OUTPUT_DIRPATH, intrinsics_refpath[:-3]+'npy')
        image_out_dirpath = \
            os.path.join(os.path.dirname(intrinsics_outpath), 'images')
        pose_out_dirpath = \
            os.path.join(os.path.dirname(intrinsics_outpath), 'poses')

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
            start_idx = skip
            offset_idx = 1

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
                paths_only))

        train_image_paths = [path(train_image_path).name for train_image_path in train_image_paths]
        test_image_paths = [path(test_image_path).name for test_image_path in test_image_paths]
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
                if image_refpath in train_image_paths:
                    train_image_outpaths.append(str(image_outpath))
                    train_pose_outpaths.append(str(pose_outpath))
                    train_sparse_depth_outpaths.append(str(sparse_depth_outpath))
                    train_validity_map_outpaths.append(str(validity_map_outpath))
                    train_ground_truth_outpaths.append(str(ground_truth_outpath))
                    train_intrinsics_outpaths.append(str(intrinsics_outpath))
                elif image_refpath in test_image_paths:
                    test_image_outpaths.append(str(image_outpath))
                    test_pose_outpaths.append(str(pose_outpath))
                    test_sparse_depth_outpaths.append(str(sparse_depth_outpath))
                    test_validity_map_outpaths.append(str(validity_map_outpath))
                    test_ground_truth_outpaths.append(str(ground_truth_outpath))
                    test_intrinsics_outpaths.append(str(intrinsics_outpath))

        n_sample = n_sample + len(pool_input)

        print('Completed processing {} examples for sequence={}'.format(
            len(pool_input), seq_dirpath))

    print('Completed processing {} examples for density={}'.format(n_sample, data_dirpath))
    orb_train_image_filepath, \
        orb_train_pose_filepath, \
        orb_train_sparse_depth_filepath, \
        orb_train_validity_map_filepath, \
        orb_train_ground_truth_filepath, \
        orb_train_intrinsics_filepath = train_filepaths

    print('Storing {} training image file paths into: {}'.format(
        len(train_image_outpaths), orb_train_image_filepath))
    data_utils.write_paths(
        orb_train_image_filepath, train_image_outpaths)

    print('Storing {} training pose file paths into: {}'.format(
        len(train_pose_outpaths), orb_train_pose_filepath))
    data_utils.write_paths(
        orb_train_pose_filepath, train_pose_outpaths)

    print('Storing {} training sparse depth file paths into: {}'.format(
        len(train_sparse_depth_outpaths), orb_train_sparse_depth_filepath))
    data_utils.write_paths(
        orb_train_sparse_depth_filepath, train_sparse_depth_outpaths)

    print('Storing {} training validity map file paths into: {}'.format(
        len(train_validity_map_outpaths), orb_train_validity_map_filepath))
    data_utils.write_paths(
        orb_train_validity_map_filepath, train_validity_map_outpaths)

    print('Storing {} training groundtruth depth file paths into: {}'.format(
        len(train_ground_truth_outpaths), orb_train_ground_truth_filepath))
    data_utils.write_paths(
        orb_train_ground_truth_filepath, train_ground_truth_outpaths)

    print('Storing {} training camera intrinsics file paths into: {}'.format(
        len(train_intrinsics_outpaths), orb_train_intrinsics_filepath))
    data_utils.write_paths(
        orb_train_intrinsics_filepath, train_intrinsics_outpaths)

    orb_test_image_filepath, \
        orb_test_pose_filepath, \
        orb_test_sparse_depth_filepath, \
        orb_test_validity_map_filepath, \
        orb_test_ground_truth_filepath, \
        orb_test_intrinsics_filepath = test_filepaths

    print('Storing {} testing image file paths into: {}'.format(
        len(test_image_outpaths), orb_test_image_filepath))
    data_utils.write_paths(
        orb_test_image_filepath, test_image_outpaths)

    print('Storing {} testing pose file paths into: {}'.format(
        len(test_pose_outpaths), orb_test_pose_filepath))
    data_utils.write_paths(
        orb_test_pose_filepath, test_pose_outpaths)

    print('Storing {} testing sparse depth file paths into: {}'.format(
        len(test_sparse_depth_outpaths), orb_test_sparse_depth_filepath))
    data_utils.write_paths(
        orb_test_sparse_depth_filepath, test_sparse_depth_outpaths)

    print('Storing {} testing validity map file paths into: {}'.format(
        len(test_validity_map_outpaths), orb_test_validity_map_filepath))
    data_utils.write_paths(
        orb_test_validity_map_filepath, test_validity_map_outpaths)

    print('Storing {} testing groundtruth depth file paths into: {}'.format(
        len(test_ground_truth_outpaths), orb_test_ground_truth_filepath))
    data_utils.write_paths(
        orb_test_ground_truth_filepath, test_ground_truth_outpaths)

    print('Storing {} testing camera intrinsics file paths into: {}'.format(
        len(test_intrinsics_outpaths), orb_test_intrinsics_filepath))
    data_utils.write_paths(
        orb_test_intrinsics_filepath, test_intrinsics_outpaths)
