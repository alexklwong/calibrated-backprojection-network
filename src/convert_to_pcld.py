import numpy as np 
import sys
import data_utils
import open3d as o3d
from matplotlib import pyplot as plt

intrinsics_file = sys.argv[1]
output_depth_file  = sys.argv[2]
try:
    image_file = sys.argv[3]
except:
    pass    
# image = o3d.io.read_image(image_file)
# depth_im/age = o3d.io.read_image(output_depth_file)

# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image, depth_image)
# o3d.visualization.draw_geometries([rgbd_image])
# exit()
K = np.loadtxt(intrinsics_file)
output_depth_image = data_utils.load_depth(output_depth_file)

image_coordinates = []
depths = []
for i in range(output_depth_image.shape[1]):
    for j in range(output_depth_image.shape[0]):
        image_coordinates.append(np.asarray([i,j,1]))
        depths.append(output_depth_image[j,i])

image_coordinates = np.asarray(image_coordinates)
image_coordinates = np.transpose(image_coordinates)
depths = np.asarray(depths)
threeD_coordinates = (np.linalg.inv(K)@image_coordinates)*depths
threeD_coordinates = threeD_coordinates.T

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(threeD_coordinates)
# o3d.io.write_point_cloud("pointcloud.pcd", pcd)
o3d.visualization.draw_geometries([pcd])
plt.plot(threeD_coordinates[:,-1])
plt.show()