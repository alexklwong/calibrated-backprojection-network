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
import numpy as np
import torch.utils.data
import data_utils

def load_pose_triplet(path):
    '''
    Load and returns 4x4 pose matrices at timt t-1,t, t+1
    '''
    poses = np.loadtxt(path)
    poses = poses.reshape(poses.shape[0], poses.shape[1]//4, 4)
    affine_row = np.asarray([0,0,0,1], dtype=poses.dtype)
    
    pose1 = poses[0,:,:].copy()
    pose1 = np.asarray(pose1, dtype=np.float32)
    pose1 = np.vstack((pose1, affine_row))
    pose0 = poses[1,:,:].copy()
    pose0 = np.asarray(pose0, dtype=np.float32)
    pose0 = np.vstack((pose0, affine_row))
    pose2 = poses[2,:,:].copy()
    pose2 = np.asarray(pose2, dtype=np.float32)
    pose2 = np.vstack((pose2, affine_row))

    return pose1, pose0, pose2

def load_image_triplet(path, normalize=True):
    '''
    Load images from image triplet

    Arg(s):
        path : str
            path to image triplet
        normalize : bool
            if set, normalize by to [0, 1] range
    Return:
        numpy[float32] : image at time t (C x H x W)
        numpy[float32] : image at time t-1 (C x H x W)
        numpy[float32] : image at time t+1 (C x H x W)
    '''

    # Load image triplet and split into images at t-1, t, t+1
    images = data_utils.load_image(
        path,
        normalize=normalize,
        data_format='CHW')

    # Split along width
    image1, image0, image2 = np.split(images, indices_or_sections=3, axis=-1)

    return image1, image0, image2

def load_depth(depth_path):
    '''
    Load depth

    Arg(s):
        depth_path : str
            path to depth map
    Return:
        numpy[float32] : depth map (1 x H x W)
    '''

    return data_utils.load_depth(depth_path, data_format='CHW')

def load_validity_map(validity_map_path):
    '''
    Load validity map

    Arg(s):
        validity_map_path : str
            path to validity map
    Returns:
        numpy[float32] : validity map (1 x H x W)
    '''

    return data_utils.load_validity_map(validity_map_path, data_format='CHW')

def random_crop(inputs, shape, intrinsics=None, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : numpy[float32]
            3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        numpy[float32] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width
    outputs = [
        T[:, y_start:y_end, x_start:x_end] for T in inputs
    ]

    if intrinsics is not None:
        # Adjust intrinsics
        intrinsics = intrinsics + [[0.0, 0.0, -x_start],
                                   [0.0, 0.0, -y_start],
                                   [0.0, 0.0, 0.0     ]]

        return outputs, intrinsics
    else:
        return outputs


class KBNetTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image at time t-1, t, and t+1
        (2) sparse depth
        (3) camera intrinsics matrix

    Arg(s):
        image_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to 3 x 3 camera intrinsics matrix
        shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 pose_paths=None,
                 shape=None,
                 random_crop_type=['none']):

        self.image_paths = image_paths
        self.pose_paths = pose_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.shape = shape
        self.do_random_crop = \
            self.shape is not None and all([x > 0 for x in self.shape])

        # Augmentation
        self.random_crop_type = random_crop_type

    def __getitem__(self, index):
        # Load image
        image1, image0, image2 = load_image_triplet(
            self.image_paths[index],
            normalize=False)

        # Load depth
        sparse_depth0 = load_depth(self.sparse_depth_paths[index])

        # Load camera intrinsics
        # print(self.intrinsics_paths[index])
        intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)

        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            [image0, image1, image2, sparse_depth0], intrinsics = random_crop(
                inputs=[image0, image1, image2, sparse_depth0],
                shape=self.shape,
                intrinsics=intrinsics,
                crop_type=self.random_crop_type)

        # Convert to float32
        image0, image1, image2, sparse_depth0, intrinsics = [
            T.astype(np.float32)
            for T in [image0, image1, image2, sparse_depth0, intrinsics]
        ]
        if self.pose_paths is None:
            return image0, image1, image2, sparse_depth0, intrinsics
        else:
            pose1, pose0, pose2 = load_pose_triplet(self.pose_paths[index])
            return image0, image1, image2, pose0, pose1, pose2, sparse_depth0, intrinsics

    def __len__(self):
        return len(self.image_paths)


class KBNetInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) camera intrinsics matrix

    Arg(s):
        image_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to 3 x 3 camera intrinsics matrix
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 use_image_triplet=True):

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.use_image_triplet = use_image_triplet

    def __getitem__(self, index):
        # Load image
        if self.use_image_triplet:
            _, image, _ = load_image_triplet(
                self.image_paths[index],
                normalize=False)
        else:
            image = data_utils.load_image(
                self.image_paths[index],
                normalize=False,
                data_format='CHW')

        # Load depth
        sparse_depth = load_depth(self.sparse_depth_paths[index])

        # Load camera intrinsics
        try:
            intrinsics = np.loadtxt(self.intrinsics_paths[index]).astype(np.float32)
        except:
            intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)

        # Convert to float32
        image, sparse_depth, intrinsics = [
            T.astype(np.float32)
            for T in [image, sparse_depth, intrinsics]
        ]

        return image, sparse_depth, intrinsics

    def __len__(self):
        return len(self.image_paths)
