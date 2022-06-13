import cv2
import numpy as np 
import glob
import os
import sys
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
sys.path.append('../')
import data_utils

N_INIT_CORNER = 1500

outer_folder = Path(sys.argv[1])
d435_image_folder = Path(str(outer_folder)+'/d435_color')
color_image_list = glob.glob(str(d435_image_folder)+"*.png")

validity_map_folder = Path(str(outer_folder)+'/validity_map')
validity_map_folder.mkdir(parents=True, exist_ok=True)

sparse_depth_folder = Path(str(outer_folder)+'/sparse_depth')
sparse_depth_folder.mkdir(parents=True, exist_ok=True)


depth_folder = Path(str(outer_folder) + '/d435_depth')
depth_image_list = glob.glob(str(depth_folder)+"*.png")

assert len(color_image_list) == len(depth_image_list), "Not same images for depth and color"

for it,image_name in enumerate(color_image_list):
    sparse_depth_filename_without_ext = Path(image_name).stem
    sparse_depth_filename = str(sparse_depth_folder) + '/' + sparse_depth_filename_without_ext + '.png'

    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    depth_image = data_utils.load_depth(depth_image_list[it])
    sparse_depth = np.zeros_like(depth_image)

    corners = cv2.cornerHarris(image, blockSize=5, ksize=3, k=0.04)
    corners = corners.ravel()
    corner_locs = np.argsort(corners)[0:N_INIT_CORNER]

    corner_map = np.zeros_like(corners)
    corner_map[corner_locs] = 1

    corner_locs = np.unravel_index(corner_locs, (image.shape[0], image.shape[1]))
    corner_locs = np.transpose(np.array([corner_locs[0], corner_locs[1]]))


    kmeans = MiniBatchKMeans(
        n_clusters=384,
        max_iter=2,
        n_init=1,
        init_size=None,
        random_state=1,
        reassignment_ratio=1e-11)
    
    kmeans.fit(corner_locs)
    corner_locs = kmeans.cluster_centers_.astype(np.uint16)

    validity_map = np.zeros_like(image).astype(np.int16)
    validity_map[corner_locs[:, 0], corner_locs[:, 1]] = 1

    sparse_depth[np.where(validity_map==1)] = dense_depth[np.where(validity_map==1)] 
    data_utils.save_depth(sparse_depth, sparse_depth_filename)

