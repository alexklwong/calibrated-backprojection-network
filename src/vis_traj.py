import numpy as np 
import cv2
from matplotlib import pyplot as plt 
import pandas as pd
import sys
from mpl_toolkits.mplot3d import Axes3D
import glob, os
from natsort import natsorted


poses_folder = sys.argv[1]
camera_pose_files = natsorted(glob.glob( os.path.join(poses_folder, "*.yml")) )

xs = []
ys = []
zs = []

for c in camera_pose_files:
    s = cv2.FileStorage()
    _ = s.open(c, cv2.FileStorage_READ)
    pnode = s.getNode('camera_pose')
    pose = pnode.mat()
    
    xs.append(pose[0,-1])
    ys.append(pose[1,-1])
    zs.append(pose[2,-1])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs,ys, zs)
plt.show()
