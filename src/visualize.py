import numpy as np 
import sys
import data_utils
import cv2

def get_depth(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print('Op: ',z[y,x],'\t', 'GT/Sparse: ', z_s[y,x])

sparse_depth_file = sys.argv[1]
dense_depth_file = sys.argv[2]
# image_file = sys.argv[3]

z_s = data_utils.load_depth(sparse_depth_file)
z_norm = (z_s - np.min(z_s)) / (np.max(z_s) - np.min(z_s))
sparse_depth_img = np.asarray(z_norm*255, np.uint8)

z = data_utils.load_depth(dense_depth_file)
z_norm = (z - np.min(z)) / (np.max(z) - np.min(z))
dense_depth_img = np.asarray(z_norm*255, np.uint8)
diff = np.abs(z_s - z)
diff[np.where(z_s==0)]=0
diff_norm = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
diff_img = np.asarray(diff_norm*255, np.uint8)




# img = cv2.imread(image_file)


cv2.imshow("sparse depth", sparse_depth_img)
cv2.namedWindow("diff")
cv2.setMouseCallback("diff", get_depth)
cv2.imshow("dense depth", dense_depth_img)
cv2.imshow('diff', diff_img)

# cv2.imshow("image", img)
cv2.waitKey(0)