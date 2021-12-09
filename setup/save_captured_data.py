import numpy as np 
import rosbag
import sys
from pathlib import Path
import cv2
from tqdm import tqdm

def save_data(messages, folder):
    folder = str(folder)
    for topic,msg,time in tqdm(messages):
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv2.imwrite(folder+'/'+str(time)+'.png', image)

bag_file = sys.argv[1]
save_folder = Path(sys.argv[2])

fisheye_folder = Path(str(save_folder) + '/fisheye_color')
fisheye_folder.mkdir(parents=True, exist_ok=True)

d435_image_folder = Path(str(save_folder)+'/d435_color')
d435_image_folder.mkdir(parents=True, exist_ok=True)

depth_folder = Path(str(save_folder) + '/d435_depth')
depth_folder.mkdir(parents=True, exist_ok=True)

print('created approp folders')

fisheye_color_topic = '/front_fisheye_camera/image_raw/compressed'   
d435_color_topic = '/torso_front_camera/color/image_raw/compressed'   
depth_topic = '/torso_front_camera/aligned_depth_to_color/image_raw/compressed'    

print('Reading bag messages')
bag = rosbag.Bag(bag_file, 'r')
fisheye_color_messages = bag.read_messages(topics=[fisheye_color_topic])    
d435_color_messages = bag.read_messages(topics=[d435_color_topic])    
depth_messages = bag.read_messages(topics=[depth_topic])

print('saving fisheye images')
save_data(fisheye_color_messages, fisheye_folder)
print('saving d435 images')
save_data(d435_color_messages, d435_image_folder)
print('saving depth images')
save_data(depth_messages, depth_folder)