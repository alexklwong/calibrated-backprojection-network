#!/usr/bin/env python
import rospy
import numpy as np 
import sys
from pathlib import Path
import tf
import cv2
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovariance, PoseWithCovarianceStamped
from cv_bridge import CvBridge
import subprocess as sp 
from PIL import Image as Im

current_pose = PoseWithCovariance()
last_pose = PoseWithCovariance()

bag_file = sys.argv[1]
try:
    player_proc = sp.Popen(['rosbag', 'play', bag_file])
except:
    rospy.logerr("Unable to play bag file. Exiting")
    exit()

save_folder = Path(sys.argv[2])

def pose_callback(msg):
    global current_pose
    current_pose = msg.pose 


class Camera():
    def __init__(self, camera):
        self.br = CvBridge()
        self.camera_name = camera[0]
        # rospy.resolve_name(self.camera_name, caller_id=None)
        self.inner_folder =  str(save_folder)+'/'+self.camera_name
        self.inner_folder_path = Path(self.inner_folder)
        try:
            self.inner_folder_path.mkdir(parents=True,exist_ok=True)
        except:
            self.inner_folder_path.mkdir(parents=True)

        self.topic = camera[1][0]
        self.compressed = camera[1][1]
        if self.compressed==1:
            self.topic = self.topic+'/compressed'
        
        if self.camera_name == 'torso_front_depth':
            self.sub = rospy.Subscriber(self.topic, Image, self.compressed_callback, queue_size=10)
        else:    
            self.sub = rospy.Subscriber(self.topic, CompressedImage, self.compressed_callback, queue_size=10)
        self.time_beg=False
        
        

    def compressed_callback(self,msg):
        global current_pose, last_pose

        current_position = np.asarray([current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z])  
        last_position = np.asarray([last_pose.pose.position.x, last_pose.pose.position.y, last_pose.pose.position.z])
        
        current_orientation = [current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w] 
        current_orientation = np.asarray(tf.transformations.euler_from_quaternion(current_orientation))

        last_orientation = [last_pose.pose.orientation.x, last_pose.pose.orientation.y, last_pose.pose.orientation.z, last_pose.pose.orientation.w] 
        last_orientation = np.asarray(tf.transformations.euler_from_quaternion(last_orientation))

        time = msg.header.stamp
        if self.time_beg == False:
            self.start_time=time
            self.time_beg=True
    
        # time = time - self.start_time    
        if (np.linalg.norm(current_position-last_position,2) >= 0.01) or np.rad2deg(np.linalg.norm(current_orientation-last_orientation))>=10:
            try:
                np_arr = None
                if self.camera_name != 'torso_front_depth':
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    cv2.imwrite(self.inner_folder+'/'+str(time)+'.jpg', image)
                else:
                    depth_image = self.br.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    depth_array = np.array(depth_image, dtype=np.uint32)
                    z = Im.fromarray(depth_array, mode='I')
                    z.save(self.inner_folder+'/'+str(time)+'.png')
                    # cv2.imwrite(self.inner_folder+'/'+str(time)+'.png', depth_image)
                    
            except:
                rospy.logwarn("Some issue with: ", self.camera_name)
                pass
            
            last_pose = current_pose

cameras = {
 'head_front':('/head_front_camera/image_raw',1),
 'front_fisheye':('/front_fisheye_camera/image_raw',1),
 'torso_front_color':('/torso_front_camera/color/image_raw',1),
 'torso_frontIR1':('/torso_front_camera/infra1/image_rect_raw',1),  
 'torso_frontIR2':('/torso_front_camera/infra2/image_rect_raw',1),
 'torso_front_depth':('/torso_front_camera/aligned_depth_to_color/image_raw',0)
 }
cameras_list = list(cameras.keys())


if __name__=='__main__':
    rospy.init_node('image_extractor')
    
    pose_sub = rospy.Subscriber('/robot_pose', PoseWithCovarianceStamped, pose_callback, queue_size=10)

    for i in range(len(cameras_list)):
        camera = cameras_list[i]
        topic, compressed =  cameras[camera]    
        cam = Camera((camera, (topic,compressed)))
        
    rospy.spin()