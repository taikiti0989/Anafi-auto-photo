#!/usr/bin/env python3

import rospy
import csv
import cv2
import math
import os
import queue
import shlex
import subprocess
import tempfile
import threading
import traceback
import time
import logging
import roslib
import sys
import numpy as np
import olympe

from std_msgs.msg import UInt8, UInt16, UInt32, Int8, Float32, String, Header, Time, Empty, Bool
from geometry_msgs.msg import PoseStamped, PointStamped, QuaternionStamped, TwistStamped, Vector3Stamped, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

from olympe.messages.drone_manager import connection_state
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, Emergency, PCMD, moveBy, CancelMoveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged, PositionChanged, SpeedChanged, AttitudeChanged, AltitudeChanged, GpsLocationChanged
from olympe.messages.ardrone3.PilotingSettings import MaxTilt, MaxDistance, MaxAltitude, NoFlyOverMaxDistance, BankedTurn
from olympe.messages.ardrone3.PilotingSettingsState import MaxTiltChanged, MaxDistanceChanged, MaxAltitudeChanged, NoFlyOverMaxDistanceChanged, BankedTurnChanged
from olympe.messages.ardrone3.SpeedSettings import MaxVerticalSpeed, MaxRotationSpeed, MaxPitchRollRotationSpeed
from olympe.messages.ardrone3.SpeedSettingsState import MaxVerticalSpeedChanged, MaxRotationSpeedChanged, MaxPitchRollRotationSpeedChanged
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
from olympe.messages.ardrone3.GPSState import NumberOfSatelliteChanged
from olympe.messages.skyctrl.CoPiloting import setPilotingSource
from olympe.messages.skyctrl.CoPilotingState import pilotingSource
from olympe.messages.skyctrl.Common import AllStates
from olympe.messages.skyctrl.CommonState import AllStatesChanged
from olympe.messages import gimbal, camera, mapper
from olympe.enums.mapper import button_event
from olympe.enums.skyctrl.CoPilotingState import PilotingSource_Source

from scipy.spatial.transform import Rotation as R

from dynamic_reconfigure.server import Server
from olympe_bridge.cfg import setAnafiConfig
from olympe_bridge.msg import PilotingCommand, CameraCommand, MoveByCommand, MoveToCommand, SkyControllerCommand

###keyboard control###  yokomatsu
# from pynput.keyboard import Listener, Key, KeyCode
import tkinter as tk
from collections import defaultdict
from enum import Enum
import threading
######################

olympe.log.update_config({"loggers": {"olympe": {"level": "ERROR"}}})

DRONE_IP = "10.202.0.1"  #シミュレーション
# DRONE_IP = "192.168.42.1" #実機用
# DRONE_IP = "192.168.11.18" #実機用2
SKYCTRL_IP = "192.168.53.1"


##################yokomatsu
pai = 3.141592
# slam_x = 0.000000
# slam_y = 0.000000
# slam_z = 0.000000
# flag = 0
##################
class Anafi(threading.Thread):
	def __init__(self):	
		if rospy.get_param("/indoor"):			
			rospy.loginfo("We are indoor")
		else:
			rospy.loginfo("We are outdoor")
					
		self.pub_image = rospy.Publisher("/anafi/image", Image, queue_size=1)
		self.pub_camerainfo = rospy.Publisher("/anafi/camera_info", CameraInfo, queue_size=1)  #yokomatsu 20220603
		self.pub_time = rospy.Publisher("/anafi/time", Time, queue_size=1)
		self.pub_attitude = rospy.Publisher("/anafi/attitude", QuaternionStamped, queue_size=1)
		self.pub_location = rospy.Publisher("/anafi/location", PointStamped, queue_size=1)
		self.pub_height = rospy.Publisher("/anafi/height", Float32, queue_size=1)
		self.pub_speed = rospy.Publisher("/anafi/speed", Vector3Stamped, queue_size=1)
		self.pub_air_speed = rospy.Publisher("/anafi/air_speed", Float32, queue_size=1)
		self.pub_link_goodput = rospy.Publisher("/anafi/link_goodput", UInt16, queue_size=1)
		self.pub_link_quality = rospy.Publisher("/anafi/link_quality", UInt8, queue_size=1)
		self.pub_wifi_rssi = rospy.Publisher("/anafi/wifi_rssi", Int8, queue_size=1)
		self.pub_battery = rospy.Publisher("/anafi/battery", UInt8, queue_size=1)
		self.pub_state = rospy.Publisher("/anafi/state", String, queue_size=1)
		self.pub_mode = rospy.Publisher("/anafi/mode", String, queue_size=1)
		self.pub_pose = rospy.Publisher("/anafi/pose", PoseStamped, queue_size=1)
		self.pub_odometry = rospy.Publisher("/anafi/odometry", Odometry, queue_size=1)
		self.pub_rpy = rospy.Publisher("/anafi/rpy", Vector3Stamped, queue_size=1)
		self.pub_skycontroller = rospy.Publisher("/skycontroller/command", SkyControllerCommand, queue_size=1)

		rospy.Subscriber("/anafi/takeoff", Empty, self.takeoff_callback)
		rospy.Subscriber("/anafi/land", Empty, self.land_callback)
		rospy.Subscriber("/anafi/emergency", Empty, self.emergency_callback)
		rospy.Subscriber("/anafi/offboard", Bool, self.offboard_callback)
		rospy.Subscriber("/anafi/cmd_rpyt", PilotingCommand, self.rpyt_callback)
		rospy.Subscriber("/anafi/cmd_moveto", MoveToCommand, self.moveTo_callback)
		rospy.Subscriber("/anafi/cmd_moveby", MoveByCommand, self.moveBy_callback)
		rospy.Subscriber("/anafi/cmd_camera", CameraCommand, self.camera_callback)

		# ########### 2022/06/21 yokomatsu slam subscribe
		# rospy.Subscriber("/bebop/pose", PoseStamped, self.orbSlamCallback)
		# ###########
		
		# Connect to the SkyController	
		if rospy.get_param("/skycontroller"):
			# rospy.loginfo("Connecting through SkyController");
			# self.drone = olympe.Drone(SKYCTRL_IP)
			rospy.loginfo("Connecting directly to Anafi");
			self.drone = olympe.Drone(DRONE_IP)
		
		# Connect to the Anafi
		else:
			rospy.loginfo("Connecting directly to Anafi");
			self.drone = olympe.Drone(DRONE_IP)
		
		# Create listener for RC events
		self.every_event_listener = EveryEventListener(self)
		
		rospy.on_shutdown(self.stop)
		
		self.srv = Server(setAnafiConfig, self.reconfigure_callback)
						
		self.connect()
				
		# To convert OpenCV images to ROS images
		self.bridge = CvBridge()
		
	def connect(self):
		self.every_event_listener.subscribe()
		
		rate = rospy.Rate(1) # 1hz
		while True:
			self.pub_state.publish("CONNECTING")
			connection = self.drone.connect()
			if getattr(connection, 'OK') == True:
				break
			if rospy.is_shutdown():
				exit()
			rate.sleep()
		
		# Connect to the SkyController	
		if rospy.get_param("/skycontroller"):
			self.pub_state.publish("CONNECTED_SKYCONTROLLER")
			rospy.loginfo("Connection to SkyController: " + getattr(connection, 'message'))
			self.switch_manual()
					
			# Connect to the drone
			while True:
				if self.drone(connection_state(state="connected", _policy="check")):
					break				
				if rospy.is_shutdown():
					exit()
				else:
					self.pub_state.publish("SERCHING_DRONE")
					rospy.loginfo_once("Connection to Anafi: " + str(self.drone.get_state(connection_state)["state"]))
				rate.sleep()
			self.pub_state.publish("CONNECTED_DRONE")			
			rospy.loginfo("Connection to Anafi: " + str(self.drone.get_state(connection_state)["state"]))
		# Connect to the Anafi
		else:
			self.pub_state.publish("CONNECTED_DRONE")
			rospy.loginfo("Connection to Anafi: " + getattr(connection, 'message'))
			# self.switch_offboard()
			self.switch_manual()
			
		self.frame_queue = queue.Queue()
		self.flush_queue_lock = threading.Lock()

		# Setup the callback functions to do some live video processing
		self.drone.set_streaming_callbacks(
			raw_cb=self.yuv_frame_cb,
			flush_raw_cb=self.flush_cb
		)
		self.drone.start_video_streaming()		
		
	def disconnect(self):
		self.pub_state.publish("DISCONNECTING")
		self.every_event_listener.unsubscribe()
		#self.drone.stop_video_streaming()
		self.drone.disconnect()
		self.pub_state.publish("DISCONNECTED")
		
	def stop(self):
		rospy.loginfo("AnafiBridge is stopping...")
		self.disconnect()
						
	def reconfigure_callback(self, config, level):
		if level == -1 or level == 1:
			self.drone(MaxTilt(config['max_tilt'])).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html?#olympe.messages.ardrone3.PilotingSettings.MaxTilt
			self.drone(MaxVerticalSpeed(config['max_vertical_speed'])).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.SpeedSettings.MaxVerticalSpeed
			self.drone(MaxRotationSpeed(config['max_yaw_rotation_speed'])).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.SpeedSettings.MaxRotationSpeed
			self.drone(MaxPitchRollRotationSpeed(config['max_pitch_roll_rotation_speed'])).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.SpeedSettings.MaxPitchRollRotationSpeed
			self.drone(MaxDistance(config['max_distance'])).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.PilotingSettings.MaxDistance
			self.drone(MaxAltitude(config['max_altitude'])).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.PilotingSettings.MaxAltitude
			self.drone(NoFlyOverMaxDistance(1)).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.PilotingSettings.NoFlyOverMaxDistance
			self.drone(BankedTurn(int(config['banked_turn']))).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.PilotingSettings.BankedTurn
			self.max_tilt = config['max_tilt']
			self.max_vertical_speed = config['max_vertical_speed']
			self.max_rotation_speed = config['max_yaw_rotation_speed']
		if level == -1 or level == 2:
			self.gimbal_frame = 'absolute' if config['gimbal_compensation'] else 'relative'
			self.drone(gimbal.set_max_speed(
				gimbal_id=0,
				yaw=0, 
				pitch=config['max_gimbal_speed'], # [1 180] (deg/s)
				roll=config['max_gimbal_speed'] # [1 180] (deg/s)
				)).wait()
		return config
		
	# This function will be called by Olympe for each decoded YUV frame.
	def yuv_frame_cb(self, yuv_frame):      
		yuv_frame.ref()
		self.frame_queue.put_nowait(yuv_frame)

	def flush_cb(self):
		with self.flush_queue_lock:
			while not self.frame_queue.empty():
				self.frame_queue.get_nowait().unref()
		return True

	def yuv_callback(self, yuv_frame):
		# Use OpenCV to convert the yuv frame to RGB
		info = yuv_frame.info() # the VideoFrame.info() dictionary contains some useful information such as the video resolution
		rospy.logdebug_throttle(10, "yuv_frame.info = " + str(info))
		cv2_cvt_color_flag = {
			olympe.PDRAW_YUV_FORMAT_I420: cv2.COLOR_YUV2BGR_I420,
			olympe.PDRAW_YUV_FORMAT_NV12: cv2.COLOR_YUV2BGR_NV12,
		}[info["yuv"]["format"]] # convert pdraw YUV flag to OpenCV YUV flag
		cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)
		##################################yokomatsu 2022/06/13
		# print(type(cv2frame))
		# print(cv2frame.dtype)
		# print("cv2frame:", cv2frame)
		
		# Publish image
		# msgmsg_header = Header()
		# msgmsg_header.frame_id = "camera_optical"
		# msgmsg_header.stamp = rospy.Time.now()
		msg_image = Image()
		msg_image = self.bridge.cv2_to_imgmsg(cv2frame, "bgr8")
		################## 試し
		# msg_image.data = cv2frame
		# msg_image.height = 720
		# msg_image.width = 1280
		# msg_image.encoding = "bgr8"
		##################
		# print(type(msg_image))
		# msg_image.header = msgmsg_header
		# self.pub_image.publish(msg_image)

		#################yokomatsu 2022/06/09
		msg_header = Header()
		msg_header.frame_id = "camera_optical"
		msg_header.stamp = rospy.Time.now()
		msg_image.header = msg_header
		msg = CameraInfo()
		msg.header = msg_header
		msg.height = 720
		msg.width = 1280
		msg.distortion_model = 'plumb_bob'
		# msg.D = [-0.065600, 0.139344, -0.001282, 0.002626, 0.000000]
		# msg.K = [899.15136, 0.0, 639.21951, 0.0, 900.71038, 359.49067, 0.0, 0.0, 1.0]
		# msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
		# msg.P = [913.26562, 0.0, 642.18948, 0.0, 0.0, 916.40509, 357.97942, 0.0, 0.0, 0.0, 1.0, 0.0]
		msg.D = [-0.036324, 0.077223, -0.003573, 0.006512, 0.000000]
		msg.K = [912.12114, 0.0, 657.29632, 0.0, 915.40109, 354.35653, 0.0, 0.0, 1.0]
		msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
		msg.P = [920.11932, 0.0, 664.68768, 0.0, 0.0, 927.3075, 351.11383, 0.0, 0.0, 0.0, 1.0, 0.0]
		msg.binning_x = 0
		msg.binning_y = 0
        # msg.roi = msg_roi
		self.pub_camerainfo.publish(msg)
		self.pub_image.publish(msg_image)
		#################
		###########################################################################

		# yuv_frame.vmeta() returns a dictionary that contains additional metadata from the drone (GPS coordinates, battery percentage, ...)
		metadata = yuv_frame.vmeta()
		rospy.logdebug_throttle(10, "yuv_frame.vmeta = " + str(metadata))
				
		if metadata[1] != None:
			header = Header()
			header.stamp = rospy.Time.now()
			header.frame_id = '/body'
		
			frame_timestamp = metadata[1]['frame_timestamp'] # timestamp [millisec]
			msg_time = Time()
			msg_time.data = frame_timestamp # secs = int(frame_timestamp//1e6), nsecs = int(frame_timestamp%1e6*1e3)
			self.pub_time.publish(msg_time)

			drone_quat = metadata[1]['drone_quat'] # attitude
			msg_attitude = QuaternionStamped()
			msg_attitude.header = header
			msg_attitude.quaternion = Quaternion(drone_quat['x'], -drone_quat['y'], -drone_quat['z'], drone_quat['w'])
			self.pub_attitude.publish(msg_attitude)
					
			location = metadata[1]['location'] # GPS location [500.0=not available] (decimal deg)
			msg_location = PointStamped()
			if location != {}:			
				msg_location.header = header
				msg_location.header.frame_id = '/world'
				msg_location.point.x = location['latitude']
				msg_location.point.y = location['longitude']
				msg_location.point.z = location['altitude']
				self.pub_location.publish(msg_location)
				
			ground_distance = metadata[1]['ground_distance'] # barometer (m)
			self.pub_height.publish(ground_distance)

			speed = metadata[1]['speed'] # opticalflow speed (m/s)
			msg_speed = Vector3Stamped()
			msg_speed.header = header
			msg_speed.header.frame_id = '/world'
			msg_speed.vector.x = speed['north']
			msg_speed.vector.y = -speed['east']
			msg_speed.vector.z = -speed['down']
			self.pub_speed.publish(msg_speed)

			air_speed = metadata[1]['air_speed'] # air speed [-1=no data, > 0] (m/s)
			self.pub_air_speed.publish(air_speed)

			link_goodput = metadata[1]['link_goodput'] # throughput of the connection (b/s)
			self.pub_link_goodput.publish(link_goodput)

			link_quality = metadata[1]['link_quality'] # [0=bad, 5=good]
			self.pub_link_quality.publish(link_quality)

			wifi_rssi = metadata[1]['wifi_rssi'] # signal strength [-100=bad, 0=good] (dBm)
			self.pub_wifi_rssi.publish(wifi_rssi)

			battery_percentage = metadata[1]['battery_percentage'] # [0=empty, 100=full]
			self.pub_battery.publish(battery_percentage)

			state = metadata[1]['state'] # ['LANDED', 'MOTOR_RAMPING', 'TAKINGOFF', 'HOWERING', 'FLYING', 'LANDING', 'EMERGENCY']
			self.pub_state.publish(state)

			mode = metadata[1]['mode'] # ['MANUAL', 'RETURN_HOME', 'FLIGHT_PLAN', 'TRACKING', 'FOLLOW_ME', 'MOVE_TO']
			self.pub_mode.publish(mode)
			
			msg_pose = PoseStamped()
			msg_pose.header = header
			msg_pose.pose.position = msg_location.point
			msg_pose.pose.position.z = ground_distance
			msg_pose.pose.orientation = msg_attitude.quaternion
			self.pub_pose.publish(msg_pose)
			
			Rot = R.from_quat([drone_quat['x'], -drone_quat['y'], -drone_quat['z'], drone_quat['w']])
			drone_rpy = Rot.as_euler('xyz')
			
			msg_odometry = Odometry()
			msg_odometry.header = header
			msg_odometry.child_frame_id = '/body'
			msg_odometry.pose.pose = msg_pose.pose
			msg_odometry.twist.twist.linear.x =  math.cos(drone_rpy[2])*msg_speed.vector.x + math.sin(drone_rpy[2])*msg_speed.vector.y
			msg_odometry.twist.twist.linear.y = -math.sin(drone_rpy[2])*msg_speed.vector.x + math.cos(drone_rpy[2])*msg_speed.vector.y
			msg_odometry.twist.twist.linear.z = msg_speed.vector.z
			self.pub_odometry.publish(msg_odometry)
			
			# log battery percentage
			if battery_percentage >= 30:
				if battery_percentage%10 == 0:
					rospy.loginfo_throttle(100, "Battery level: " + str(battery_percentage) + "%")
			else:
				if battery_percentage >= 20:
					rospy.logwarn_throttle(10, "Low battery: " + str(battery_percentage) + "%")
				else:
					if battery_percentage >= 10:
						rospy.logerr_throttle(1, "Critical battery: " + str(battery_percentage) + "%")
					else:
						rospy.logfatal_throttle(0.1, "Empty battery: " + str(battery_percentage) + "%")		
					
			# log signal strength
			if wifi_rssi <= -60:
				if wifi_rssi >= -70:
					rospy.loginfo_throttle(100, "Signal strength: " + str(wifi_rssi) + "dBm")
				else:
					if wifi_rssi >= -80:
						rospy.logwarn_throttle(10, "Weak signal: " + str(wifi_rssi) + "dBm")
					else:
						if wifi_rssi >= -90:
							rospy.logerr_throttle(1, "Unreliable signal:" + str(wifi_rssi) + "dBm")
						else:
							rospy.logfatal_throttle(0.1, "Unusable signal: " + str(wifi_rssi) + "dBm")
		else:
			rospy.logwarn("Packet lost!")

	######################################################################yokomatsu 2022/06/09
	# def parse_yaml(filename):
    # 	with open(filename, 'r')
   	# 	calib_data = yaml.load(stream)
    # 	cam_info = CameraInfo()
    # 	cam_info.width = calib_data['image_width']
    # 	cam_info.height = calib_data['image_height']
    # 	cam_info.K = calib_data['camera_matrix']['data']
    # 	cam_info.D = calib_data['distortion_coefficients']['data']
    # 	cam_info.R = calib_data['rectification_matrix']['data']
    # 	cam_info.P = calib_data['projection_matrix']['data']
    # 	cam_info.distortion_model = calib_data['distortion_model']
    # 	cam_info.binning_x = calib_data['binning_x']
    # 	cam_info.binning_y = calib_data['binning_y']
    # 	cam_info.roi.x_offset = calib_data['roi']['x_offset']
    # 	cam_info.roi.y_offset = calib_data['roi']['y_offset']
    # 	cam_info.roi.height = calib_data['roi']['height']
    # 	cam_info.roi.width = calib_data['roi']['width']
    # 	cam_info.roi.do_rectify = calib_data['roi']['do_rectify']
    
    # 	return cam_info
	#######################################################################################

	################################################# 2022/06/21 yokomatsu slam
	def orbSlamCallback(self, msg):
		global slam_x, slam_y, slam_z
		slam_x = msg.pose.position.x
		slam_y = msg.pose.position.y
		slam_z = msg.pose.position.z
		# print("slam_x", slam_x)
		# print("slam_y", slam_y)
		# print("slam_z", slam_z)
        # slam_position_csv << slam_x <<","<< slam_y <<","<< slam_z <<endl;
        # ROS_WARN("slam pose output");
        
        # position.x = slam_x;
        # position.y = slam_y;
		# double q0,q1,q2,q3
		q0 = msg.pose.orientation.x 
		q1 = msg.pose.orientation.y
		q2 = msg.pose.orientation.z
		q3 = msg.pose.orientation.w
		siny_cosp = +2.0 * (q3 * q0 + q2 * q1)
		cosy_cosp = +1.0 - 2.0 * (q3 * q3 + q2 * q2)
		slam_yaw = math.atan2(siny_cosp, cosy_cosp)
        # cout<<"yaw: "<<slam_yaw<<endl;
        # double yaw = marker_yaw*180/pai;
		slam_roll = math.atan2(2*(q0*q1+q2*q3),1-2*(q1*q1+q2*q2))
		roll = slam_roll*180/pai
        # cout<<"roll: "<<slam_roll<<endl;
		marker_pitch = math.asin(2*(q0*q2-q3*q1))
		pitch = marker_pitch*180/pai
        # circle(position_image,position,5,Scalar(0,0,0));
        # cv::imshow("map",position_image);
        # cv::waitKey(100);
        # cout<<"pitch: "<<marker_pitch<<endl
        # fout_data_marker_x<<slam_x<<endl
        # fout_data_marker_y<<slam_y<<endl
        # fout_data_marker_z<<slam_z<<endl
        # fout_data_marker_yaw<<slam_roll<<endl
		# rospy.loginfo("22222222")
		return slam_x, slam_y, slam_z
	#################################################
		
	def takeoff_callback(self, msg):		
		self.drone(TakeOff() >> FlyingStateChanged(state="hovering", _timeout=10)).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.TakeOff
		rospy.logwarn("Takeoff")

	def land_callback(self, msg):		
		self.drone(Landing()).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.Landing
		rospy.loginfo("Land")

	def emergency_callback(self, msg):		
		self.drone(Emergency()).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.Emergency
		rospy.logfatal("Emergency!!!")
		
	def offboard_callback(self, msg):
		if msg.data == False:	
			self.switch_manual()
		else:
			self.switch_offboard()
						
	def rpyt_callback(self, msg):
		self.drone(PCMD( # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.PCMD
			flag=1,
			roll=int(self.bound_percentage(msg.roll/self.max_tilt*100)), # roll [-100, 100] (% of max tilt)
			pitch=int(self.bound_percentage(msg.pitch/self.max_tilt*100)), # pitch [-100, 100] (% of max tilt)
			yaw=int(self.bound_percentage(-msg.yaw/self.max_rotation_speed*100)), # yaw rate [-100, 100] (% of max yaw rate)
			gaz=int(self.bound_percentage(msg.gaz/self.max_vertical_speed*100)), # vertical speed [-100, 100] (% of max vertical speed)
			timestampAndSeqNum=0)) # for debug only

	# TODO: NOT USED YET	
	def moveBy_callback(self, msg):		
		assert self.drone(moveBy( # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.moveBy
			dX=msg.dx, # displacement along the front axis (m)
			dY=msg.dy, # displacement along the right axis (m)
			dZ=msg.dz, # displacement along the down axis (m)
			dPsi=msg.dyaw # rotation of heading (rad)
			) >> FlyingStateChanged(state="hovering", _timeout=1)
		).wait().success()
	
	# TODO: NOT USED YET	
	def moveTo_callback(self, msg):		
		assert self.drone(moveTo( # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.moveTo
			latitude=msg.latitude, # latitude (degrees)
			longitude=msg.longitude, # longitude (degrees)
			altitude=msg.altitude, # altitude (m)
			heading=msg.heading, # heading relative to the North (degrees)
			orientation_mode=HEADING_START # orientation mode {TO_TARGET, HEADING_START, HEADING_DURING}
			) >> FlyingStateChanged(state="hovering", _timeout=5)
		).wait().success()

	def camera_callback(self, msg):
		if msg.action & 0b001: # take picture
			self.drone(camera.take_photo(cam_id=0)) # https://developer.parrot.com/docs/olympe/arsdkng_camera.html#olympe.messages.camera.take_photo
		if msg.action & 0b010: # start recording
			self.drone(camera.start_recording(cam_id=0)).wait() # https://developer.parrot.com/docs/olympe/arsdkng_camera.html#olympe.messages.camera.start_recording
		if msg.action & 0b100: # stop recording
			self.drone(camera.stop_recording(cam_id=0)).wait() # https://developer.parrot.com/docs/olympe/arsdkng_camera.html#olympe.messages.camera.stop_recording
	
		self.drone(gimbal.set_target( # https://developer.parrot.com/docs/olympe/arsdkng_gimbal.html#olympe.messages.gimbal.set_target
			gimbal_id=0,
			control_mode='position', # {'position', 'velocity'}
			yaw_frame_of_reference='none',
			yaw=0.0,
			pitch_frame_of_reference=self.gimbal_frame, # {'absolute', 'relative', 'none'}
			pitch=msg.pitch,
			roll_frame_of_reference=self.gimbal_frame, # {'absolute', 'relative', 'none'}
			roll=msg.roll))
			
		self.drone(camera.set_zoom_target( # https://developer.parrot.com/docs/olympe/arsdkng_camera.html#olympe.messages.camera.set_zoom_target
			cam_id=0,
			control_mode='level', # {'level', 'velocity'}
			target=msg.zoom)) # [1, 3]

	def switch_manual(self):
		msg_rpyt = SkyControllerCommand()
		msg_rpyt.header.stamp = rospy.Time.now()
		msg_rpyt.header.frame_id = '/body'
		self.pub_skycontroller.publish(msg_rpyt)
		
		# button: 	0 = RTL, 1 = takeoff/land, 2 = back left, 3 = back right
		self.drone(mapper.grab(buttons=(0<<0|0<<1|0<<2|1<<3), axes=0)).wait() # bitfields
		self.drone(setPilotingSource(source="SkyController")).wait()
		rospy.loginfo("Control: Manual")
			
	def switch_offboard(self):
		# button: 	0 = RTL, 1 = takeoff/land, 2 = back left, 3 = back right
		# axis: 	0 = yaw, 1 = trottle, 2 = roll, 3 = pithch, 4 = camera, 5 = zoom
		if self.drone.get_state(pilotingSource)["source"] == PilotingSource_Source.SkyController:
			self.drone(mapper.grab(buttons=(1<<0|0<<1|1<<2|1<<3), axes=(1<<0|1<<1|1<<2|1<<3|0<<4|0<<5))) # bitfields
			self.drone(setPilotingSource(source="Controller")).wait()
			rospy.loginfo("Control: Offboard")
		else:
			self.switch_manual()
			
	def bound(self, value, value_min, value_max):
		return min(max(value, value_min), value_max)
		
	def bound_percentage(self, value):
		return self.bound(value, -100, 100)

	def run(self): 
		rate = rospy.Rate(100) # 100hz
		
		rospy.logdebug('MaxTilt = %f [%f, %f]', self.drone.get_state(MaxTiltChanged)["current"], self.drone.get_state(MaxTiltChanged)["min"], self.drone.get_state(MaxTiltChanged)["max"])
		rospy.logdebug('MaxVerticalSpeed = %f [%f, %f]', self.drone.get_state(MaxVerticalSpeedChanged)["current"], self.drone.get_state(MaxVerticalSpeedChanged)["min"], self.drone.get_state(MaxVerticalSpeedChanged)["max"])
		rospy.logdebug('MaxRotationSpeed = %f [%f, %f]', self.drone.get_state(MaxRotationSpeedChanged)["current"], self.drone.get_state(MaxRotationSpeedChanged)["min"], self.drone.get_state(MaxRotationSpeedChanged)["max"])
		rospy.logdebug('MaxPitchRollRotationSpeed = %f [%f, %f]', self.drone.get_state(MaxPitchRollRotationSpeedChanged)["current"], self.drone.get_state(MaxPitchRollRotationSpeedChanged)["min"], self.drone.get_state(MaxPitchRollRotationSpeedChanged)["max"])
		rospy.logdebug('MaxDistance = %f [%f, %f]', self.drone.get_state(MaxDistanceChanged)["current"], self.drone.get_state(MaxDistanceChanged)["min"], self.drone.get_state(MaxDistanceChanged)["max"])
		rospy.logdebug('MaxAltitude = %f [%f, %f]', self.drone.get_state(MaxAltitudeChanged)["current"], self.drone.get_state(MaxAltitudeChanged)["min"], self.drone.get_state(MaxAltitudeChanged)["max"])
		rospy.logdebug('NoFlyOverMaxDistance = %i', self.drone.get_state(NoFlyOverMaxDistanceChanged)["shouldNotFlyOver"])
		rospy.logdebug('BankedTurn = %i', self.drone.get_state(BankedTurnChanged)["state"])
		maxTiltAction = self.drone(MaxTilt(40, _timeout=1)).wait()
		# control = KeyboardCtrl()
		# while not control.quit():
		# 	if control.takeoff():
		# 		self.drone(TakeOff())
		# 	elif control.landing():
		# 		self.drone(Landing())
		# 	if control.has_piloting_cmd():
		# 		self.drone(
		# 			PCMD(
		# 				1,
		# 				control.roll(),
		# 				control.pitch(),
		# 				control.yaw(),
		# 				control.throttle(),
		# 				timestampAndSeqNum=0,
		# 			)
		# 		)
		# 	else:
		# 		self.drone(PCMD(0, 0, 0, 0, 0, timestampAndSeqNum=0))
		# 	time.sleep(0.05)
		# anafi.takeoff_callback()
		# self.drone(TakeOff() >> FlyingStateChanged(state="hovering", _timeout=10)).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.TakeOff
		# rospy.logwarn("Takeoff")
		# self.drone(Landing()).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.Landing
		# rospy.loginfo("Land")
	
		while not rospy.is_shutdown():
			def process():
				while True:
					with self.flush_queue_lock:
						try:					
							yuv_frame = self.frame_queue.get(timeout=0.01)
						except queue.Empty:
							continue
				
						try:
							self.yuv_callback(yuv_frame)
						except Exception:
							# Continue popping frame from the queue even if it fails to show one frame
							traceback.print_exc()
							continue
						finally:
							# Unref the yuv frame to avoid starving the video buffer pool
							yuv_frame.unref()
					rate.sleep()
			threadA = threading.Thread(target=process)
			threadA.start()

			def btn_up():       #slam waypointでmoveByの動作検証
				while flag1 == 2:
					global move_flag
					if move_flag == 1:
						print("move_flag", move_flag)
						print("slam x:", slam_x)
						print("slam y:", slam_y)
						print("slam z:", slam_z)
						waypoint_x = 0.0
						waypoint_y = 0.0
						# waypoint_z = 1.0
						dis_x = (waypoint_x - slam_x)
						print("dis_x", dis_x)
						dis_y = (waypoint_y - slam_y)
						print("dis_y", dis_y)
						# dis_z = (waypoint_z - slam_z) ** 2
						# print("dis_z", dis_z)
						self.drone(moveBy(
							dis_y,  #前方向に移動
							dis_x,  #横方向に移動
							0,
							0,
							_timeout=10,
							_no_expect=False,
							_float_tol=(1e-07, 1e-09)
						))
						print('nyuuryokuukwtuketa')
						disdis = np.sqrt(dis_x ** 2 + dis_y ** 2)
						print("disdis", disdis)
						# disdisdis = np.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
						# print("disdisdis", disdisdis)
						# time.sleep(5)
						if disdis < 0.1:
							print("ok")
							self.drone(Landing())
							# move_flag = 2
						#############################################################
					if move_flag == 2:
						print("move_flag", move_flag)
						print("slam x:", slam_x)
						print("slam y:", slam_y)
						print("slam z:", slam_z)
						waypoint_x = -1.0
						waypoint_y = 1.0
						# waypoint_z = 1.0
						dis_x = (waypoint_x - slam_x)
						print("dis_x", dis_x)
						dis_y = (waypoint_y - slam_y)
						print("dis_y", dis_y)
						# dis_z = (waypoint_z - slam_z) ** 2
						# print("dis_z", dis_z)
						self.drone(moveBy(
							dis_y,  #前方向に移動
							dis_x,  #横方向に移動
							0,
							0,
							_timeout=10,
							_no_expect=False,
							_float_tol=(1e-07, 1e-09)
						))
						disdis = np.sqrt(dis_x ** 2 + dis_y ** 2)
						print("disdis", disdis)
						# disdisdis = np.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
						# print("disdisdis", disdisdis)
						# time.sleep(5)
						if disdis < 0.1:
							print("ok")
							move_flag = 3
					##################################################################
					if move_flag == 3:
						print("move_flag", move_flag)
						print("slam x:", slam_x)
						print("slam y:", slam_y)
						print("slam z:", slam_z)
						waypoint_x = -1.0
						waypoint_y = -1.0
						# waypoint_z = 1.0
						dis_x = (waypoint_x - slam_x)
						print("dis_x", dis_x)
						dis_y = (waypoint_y - slam_y)
						print("dis_y", dis_y)
						# dis_z = (waypoint_z - slam_z) ** 2
						# print("dis_z", dis_z)
						self.drone(moveBy(
							dis_y,  #前方向に移動
							dis_x,  #横方向に移動
							0,
							0,
							_timeout=10,
							_no_expect=False,
							_float_tol=(1e-07, 1e-09)
						))
						disdis = np.sqrt(dis_x ** 2 + dis_y ** 2)
						print("disdis", disdis)
						# disdisdis = np.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
						# print("disdisdis", disdisdis)
						# time.sleep(5)
						if disdis < 0.1:
							print("ok")
							move_flag = 4
					##################################################################
					if move_flag == 4:
						print("move_flag", move_flag)
						print("slam x:", slam_x)
						print("slam y:", slam_y)
						print("slam z:", slam_z)
						waypoint_x = 1.0
						waypoint_y = -1.0
						# waypoint_z = 1.0
						dis_x = (waypoint_x - slam_x)
						print("dis_x", dis_x)
						dis_y = (waypoint_y - slam_y)
						print("dis_y", dis_y)
						# dis_z = (waypoint_z - slam_z) ** 2
						# print("dis_z", dis_z)
						self.drone(moveBy(
							dis_y,  #前方向に移動
							dis_x,  #横方向に移動
							0,
							0,
							_timeout=10,
							_no_expect=False,
							_float_tol=(1e-07, 1e-09)
						))
						disdis = np.sqrt(dis_x ** 2 + dis_y ** 2)
						print("disdis", disdis)
						# disdisdis = np.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
						# print("disdisdis", disdisdis)
						# time.sleep(5)
						if disdis < 0.1:
							print("ok")
							move_flag = 5
					##################################################################
					if move_flag == 5:
						print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
			# def btn_up():       #緊急停止ボタン、動作確認用
			# 	while flag1 == 2:
			# 		self.drone(moveBy(
			# 			100,
			# 			0,
			# 			0,
			# 			0,
			# 			_timeout=10,
			# 			_no_expect=False,
			# 			_float_tol=(1e-07, 1e-09)
			# 		))

			connection = self.drone.connection_state()
			rospy.loginfo("111");
			# control = KeyboardCtrl()
			root = tk.Tk()
			rospy.loginfo("12345");
			root.title(u"keyboard controll")
			def key_event(e):
				global flag1
				global move_flag
				key = e.keysym
				print("入力したキー:", e.keysym)
				if key == "t":
					self.drone(TakeOff())
				if key == "l":
					self.drone(Landing())
				#ポイント移動X軸前
				if key == "w":
					self.drone(PCMD(
						1,
						0,
						10,
						0,
						0,
						timestampAndSeqNum=0,
					))
				if key == "s":
					self.drone(PCMD(
						1,
						0,
						-10,
						0,
						0,
						timestampAndSeqNum=0,
					))
				#ポイント移動Y軸
				if key == "a":
					self.drone(PCMD(
						1,
						-50,
						0,
						0,
						0,
						timestampAndSeqNum=0,
					))
				if key == "d":
					self.drone(PCMD(
						1,
						50,
						0,
						0,
						0,
						timestampAndSeqNum=0,
					))
				if key == "e":
					self.drone(PCMD(
						1,
						0,
						0,
						50,
						0,
						timestampAndSeqNum=0,
					))
				if key == "q":
					self.drone(PCMD(
						1,
						0,
						0,
						-50,
						0,
						timestampAndSeqNum=0,
					))
				#ポイント移動Z軸
				if key == "Down":
					self.drone(PCMD(
						1,
						0,
						0,
						0,
						-50,
						timestampAndSeqNum=0,
					))
				if key == "Up":
					self.drone(PCMD(
						1,
						0,
						0,
						0,
						50,
						timestampAndSeqNum=0,
					))
				if key == "g":  #カメラ角度移動
					self.drone(gimbal.set_target( # https://developer.parrot.com/docs/olympe/arsdkng_gimbal.html#olympe.messages.gimbal.set_target
						gimbal_id=0,
						control_mode='position', # {'position', 'velocity'}
						yaw_frame_of_reference='none',
						yaw=0.0,
						pitch_frame_of_reference='absolute', # {'absolute', 'relative', 'none'}
						pitch=0,
						roll_frame_of_reference=self.gimbal_frame, # {'absolute', 'relative', 'none'}
						roll=0
						)
					)
				if key == "h":
					self.drone(gimbal.set_target( # https://developer.parrot.com/docs/olympe/arsdkng_gimbal.html#olympe.messages.gimbal.set_target
						gimbal_id=0,
						control_mode='position', # {'position', 'velocity'}
						yaw_frame_of_reference='none',
						yaw=0.0,
						pitch_frame_of_reference='absolute', # {'absolute', 'relative', 'none'}
						pitch=-40,
						roll_frame_of_reference=self.gimbal_frame, # {'absolute', 'relative', 'none'}
						roll=0
						)
					)
################################################## 2022/06/23  slam座標における原点に移動
				if key == "7":
					threadB = threading.Thread(target=btn_up)
					threadB.start()
						# print("slam x:", slam_x)
						# print("slam y:", slam_y)
						# print("slam z:", slam_z)
						# dis_x = (0 - slam_x) ** 2
						# print("dis_x", dis_x)
						# dis_y = (0 - slam_y) ** 2
						# print("dis_y", dis_y)
						# disdis = np.sqrt(dis_x + dis_y)
						# print("disdis", disdis)
						# if disdis < 0.1:
						# 	print("ok")
						# 	flag1 = 1
				if key == "8":
					flag1 = 10
					print("flag1:", flag1)
					self.drone(CancelMoveBy(_timeout = 10, _no_expect = False, _float_tol = (1e-07, 1e-09)))
				if key == "9":
					flag1 = 2
					move_flag = 1
					print("flag1:", flag1)
					print("move_flag:", move_flag)					
				if key == "b":
					self.drone(moveBy(
						slam_x,
						slam_y,
						0,
						0,
						_timeout=10,
						_no_expect=False,
						_float_tol=(1e-07, 1e-09)
					))
				if key == "m": 
					self.drone(CancelMoveBy(_timeout = 10, _no_expect = False, _float_tol = (1e-07, 1e-09)))

#######################################################
			root.bind("<KeyPress>", key_event)
			root.mainloop()

####################ここの内容を変える####################################			
			# while not control.quit():
			# 	if control.takeoff():
			# 		self.drone(TakeOff())
			# 	elif control.landing():
			# 		self.drone(Landing())
			# 	if control.has_piloting_cmd():
			# 		self.drone(
			# 			PCMD(
			# 				1,
			# 				control.roll(),
			# 				control.pitch(),
			# 				control.yaw(),
			# 				control.throttle(),
			# 				timestampAndSeqNum=0,
			# 			)
			# 		)
			# 	else:
			# 		self.drone(PCMD(0, 0, 0, 0, 0, timestampAndSeqNum=0))
			# 	time.sleep(0.05)
#######################################################
			if getattr(connection, 'OK') == False:
				rospy.logfatal(getattr(connection, 'message'))
				rospy.loginfo("222");
				self.disconnect()
				self.connect()
			
			# SLOW -- 5Hz			
			#attitude = self.drone.get_state(AttitudeChanged) # attitude
			#rospy.loginfo("Attitude: " + str(attitude))
			#msg_rpy = Vector3Stamped()
			#msg_rpy.header.seq = self.seq
			#msg_rpy.header.stamp = rospy.Time.now()
			#msg_rpy.header.frame_id = '/world'
			#msg_rpy.vector.x = attitude['roll']/math.pi*180
			#msg_rpy.vector.y = -attitude['pitch']/math.pi*180
			#msg_rpy.vector.z = -attitude['yaw']/math.pi*180
			#self.pub_rpy.publish(msg_rpy)

class EveryEventListener(olympe.EventListener):
	def __init__(self, anafi):
		self.anafi = anafi
				
		self.msg_rpyt = SkyControllerCommand()
		
		super().__init__(anafi.drone)

	def print_event(self, event): # Serializes an event object and truncates the result if necessary before printing it
		if isinstance(event, olympe.ArsdkMessageEvent):
			max_args_size = 200
			args = str(event.args)
			args = (args[: max_args_size - 3] + "...") if len(args) > max_args_size else args
			rospy.logdebug("{}({})".format(event.message.fullName, args))
		else:
			rospy.logdebug(str(event))

	# RC buttons listener     
	@olympe.listen_event(mapper.grab_button_event()) # https://developer.parrot.com/docs/olympe/arsdkng_mapper.html#olympe.messages.mapper.grab_button_event
	def on_grab_button_event(self, event, scheduler):
		self.print_event(event)
		# button: 	0 = RTL, 1 = takeoff/land, 2 = back left, 3 = back right
		# axis_button:	4 = max CCW yaw, 5 = max CW yaw, 6 = max trottle, 7 = min trottle
		# 		8 = min roll, 9 = max roll, 10 = min pitch, 11 = max pitch
		# 		12 = max camera down, 13 = max camera up, 14 = min zoom, 15 = max zoom
		if event.args["event"] == button_event.press:
			if event.args["button"] == 0: # RTL
				self.anafi.drone(Emergency()).wait()
				rospy.logfatal("Emergency!!!")
				return
			if event.args["button"] == 2: # left back button
				self.anafi.switch_manual()
				return
			if event.args["button"] == 3: # right back button
				self.anafi.switch_offboard()
				self.msg_rpyt = SkyControllerCommand()
				return
        
      	# RC axis listener
	@olympe.listen_event(mapper.grab_axis_event()) # https://developer.parrot.com/docs/olympe/arsdkng_mapper.html#olympe.messages.mapper.grab_axis_event
	def on_grab_axis_event(self, event, scheduler):	
		# axis: 	0 = yaw, 1 = z, 2 = y, 3 = x, 4 = camera, 5 = zoom
		if event.args["axis"] == 0: # yaw
			self.msg_rpyt.yaw = event.args["value"]
		if event.args["axis"] == 1: # z
			self.msg_rpyt.z = event.args["value"]
		if event.args["axis"] == 2: # y/pitch
			self.msg_rpyt.y = event.args["value"]
		if event.args["axis"] == 3: # x/roll
			self.msg_rpyt.x = event.args["value"]
			
		self.msg_rpyt.header.stamp = rospy.Time.now()
		self.msg_rpyt.header.frame_id = '/body'
		self.anafi.pub_skycontroller.publish(self.msg_rpyt)
		          
      	# All other events
	@olympe.listen_event()
	def default(self, event, scheduler):
		#self.print_event(event)
		pass


###keyboard class#######
class Ctrl(Enum):
	(
		QUIT,
		TAKEOFF,
		LANDING,
		MOVE_LEFT,
		MOVE_RIGHT,
		MOVE_FORWARD,
		MOVE_BACKWARD,
		MOVE_UP,
		MOVE_DOWN,
		TURN_LEFT,
		TURN_RIGHT,
	) = range(11)


QWERTY_CTRL_KEYS = {
	Ctrl.QUIT: "Escape",
	Ctrl.TAKEOFF: "t",
	Ctrl.LANDING: "l",
	Ctrl.MOVE_LEFT: "a",
	Ctrl.MOVE_RIGHT: "d",
	Ctrl.MOVE_FORWARD: "w",
	Ctrl.MOVE_BACKWARD: "s",
	Ctrl.MOVE_UP: "Up",
	Ctrl.MOVE_DOWN: "Down",
	Ctrl.TURN_LEFT: "Left",
	Ctrl.TURN_RIGHT: "Right",
}

AZERTY_CTRL_KEYS = QWERTY_CTRL_KEYS.copy()
AZERTY_CTRL_KEYS.update(
	{
		Ctrl.MOVE_LEFT: "q",
		Ctrl.MOVE_RIGHT: "d",
		Ctrl.MOVE_FORWARD: "z",
		Ctrl.MOVE_BACKWARD: "s",	
	}
)


# class KeyboardCtrl(Listener):
class KeyboardCtrl():
	def __init__(self, ctrl_keys=None):
		self._ctrl_keys = self._get_ctrl_keys(ctrl_keys)
		self._key_pressed = defaultdict(lambda: False)
		self._last_action_ts = defaultdict(lambda: 0.0)
		super().__init__(on_press=self._on_press, on_release=self._on_release)
		# self.start()

	# def _on_press(self, key):
	# 	if isinstance(key, KeyCode):
	# 		self._key_pressed[key.char] = True
	# 	elif isinstance(key, Key):
	# 		self._key_pressed[key] = True
	# 	if self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]:
	# 		return False
	# 	else:
	# 		return True

	# def _on_release(self, key):
	# 	if isinstance(key, KeyCode):
	# 		self._key_pressed[key.char] = False
	# 	elif isinstance(key, Key):
	# 		self._key_pressed[key] = False
	# 	return True
	
	# def quit(self):
	# 	return not self.running or self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]

	def _axis(self, left_key, right_key):
		return 100 * (
			int(self._key_pressed[right_key]) - int(self._key_pressed[left_key])
		)

	def roll(self):
		return self._axis(
			self._ctrl_keys[Ctrl.MOVE_LEFT],
			self._ctrl_keys[Ctrl.MOVE_RIGHT]
		)
	
	def pitch(self):
		return self._axis(
			self._ctrl_keys[Ctrl.MOVE_BACKWARD],
			self._ctrl_keys[Ctrl.MOVE_FORWARD]
		)

	def yaw(self):
		return self._axis(
			self._ctrl_keys[Ctrl.TURN_LEFT],
			self._ctrl_keys[Ctrl.TURN_RIGHT]
        )

	def throttle(self):
		return self._axis(
			self._ctrl_keys[Ctrl.MOVE_DOWN],
			self._ctrl_keys[Ctrl.MOVE_UP]
		)

	def has_piloting_cmd(self):
		return (
			bool(self.roll())
			or bool(self.pitch())
			or bool(self.yaw())
			or bool(self.throttle())
		)

	def _rate_limit_cmd(self, ctrl, delay):
		now = time.time()
		if self._last_action_ts[ctrl] > (now - delay):
			return False
		elif self._key_pressed[self._ctrl_keys[ctrl]]:
			self._last_action_ts[ctrl] = now
			return True
		else:
			return False

	def takeoff(self):
		return self._rate_limit_cmd(Ctrl.TAKEOFF, 2.0)

	def landing(self):
		return self._rate_limit_cmd(Ctrl.LANDING, 2.0)

	def _get_ctrl_keys(self, ctrl_keys):
		# Get the default ctrl keys based on the current keyboard layout:
		if ctrl_keys is None:
			ctrl_keys = QWERTY_CTRL_KEYS
			try:
				# Olympe currently only support Linux
				# and the following only works on *nix/X11...
				keyboard_variant = (
					subprocess.check_output(
						"setxkbmap -query | grep 'variant:'|"
						"cut -d ':' -f2 | tr -d ' '",
						shell=True,
					)
					.decode()
					.strip()
				)
			except subprocess.CalledProcessError:
				pass
			else:
				if keyboard_variant == "azerty":
					ctrl_keys = AZERTY_CTRL_KEYS
		return ctrl_keys
########################

if __name__ == '__main__':
	rospy.init_node('anafi_bridge', anonymous = False)
	rospy.loginfo("AnafiBridge is running...")
	anafi = Anafi()
	########### 2022/06/21 yokomatsu slam subscribe
	sub_slam = rospy.Subscriber("/bebop/pose", PoseStamped, anafi.orbSlamCallback)
	###########	
	anafi.drone(gimbal.set_target( # https://developer.parrot.com/docs/olympe/arsdkng_gimbal.html#olympe.messages.gimbal.set_target
						gimbal_id=0,
						control_mode='position', # {'position', 'velocity'}
						yaw_frame_of_reference='none',
						yaw=0.0,
						pitch_frame_of_reference='absolute', # {'absolute', 'relative', 'none'}
						pitch=-40,
						roll_frame_of_reference=anafi.gimbal_frame, # {'absolute', 'relative', 'none'}
						roll=0,
						)
					)
	try:
		anafi.run()
	except rospy.ROSInterruptException:
		traceback.print_exc()
		pass
