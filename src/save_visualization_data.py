import numpy as np 
import sys
import data_utils
import cv2
from pathlib import Path as path

def get_depth(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print('Op: ', output_depth[y,x],'\t', 'GT/Sparse: ', gt_depth[y,x])

def load_normalized_depth(file):
    z = data_utils.load_depth(file)
    z_norm = (z - np.min(z)) / (np.max(z) - np.min(z))
    depth_img = np.asarray(z_norm*255, np.uint8)
    return z,z_norm,depth_img

full_filepath = path(sys.argv[1])
intrinsics_mat = None
try:
    intrinsics_matpath = sys.argv[2]
    intrinsics_mat = np.load(intrinsics_matpath).astype(np.float32)
except:
    pass

filename = full_filepath.name
outer_dirpath = full_filepath.parent.parent

sparse_depth_file = str(outer_dirpath)+'/sparse_depth/'+filename
output_dense_depth_file = str(outer_dirpath)+'/output_depth/'+filename
gt_depth_file = str(outer_dirpath)+'/ground_truth/'+filename
image_file = str(outer_dirpath)+'/image/'+filename

sparse_depth, sparse_depth_norm, sparse_depth_img = load_normalized_depth(sparse_depth_file)
output_depth, output_depth_norm, output_depth_img = load_normalized_depth(output_dense_depth_file)
gt_depth, gt_depth_norm, gt_depth_img = load_normalized_depth(gt_depth_file)
img = cv2.imread(image_file)

diff = np.sqrt( (gt_depth - output_depth)*(gt_depth - output_depth)  )
diff[np.where(gt_depth==0)]=0
diff_norm = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
diff_img = np.asarray(diff_norm*255, np.uint8)

cv2.imwrite("/home/rakshith/CTU/Sem_4/thesis/ctuthesis/output_visualizations/image.png", img)
cv2.imwrite("/home/rakshith/CTU/Sem_4/thesis/ctuthesis/output_visualizations/sparse_depth.png", sparse_depth_img)
cv2.imwrite("/home/rakshith/CTU/Sem_4/thesis/ctuthesis/output_visualizations/dense_depth.png", output_depth_img)
cv2.imwrite("/home/rakshith/CTU/Sem_4/thesis/ctuthesis/output_visualizations/gt_depth.png", gt_depth_img)
cv2.imwrite("/home/rakshith/CTU/Sem_4/thesis/ctuthesis/output_visualizations/diff_img.png", diff_img)
cv2.imshow("sparse depth", sparse_depth_img)
cv2.namedWindow("dense depth")
cv2.setMouseCallback("dense depth", get_depth)
cv2.imshow("dense depth", output_depth_img)
cv2.imshow('diff', diff_img)

cv2.imshow("image", img)
cv2.waitKey(0)