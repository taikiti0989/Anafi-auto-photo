#!/usr/bin/env python3

from ast import If
from ftplib import error_perm
from sqlite3 import Timestamp
from this import d
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
from olympe_bridge.msg import PilotingCommand, CameraCommand, MoveByCommand, MoveToCommand, SkyControllerCommand #, control_input

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###keyboard control###  yokomatsu
# from pynput.keyboard import Listener, Key, KeyCode
import tkinter as tk
from collections import defaultdict
from enum import Enum
import threading
######################

olympe.log.update_config({"loggers": {"olympe": {"level": "ERROR"}}})

#DRONE_IP = "10.202.0.1"  #シミュレーション
DRONE_IP = "192.168.42.1" #実機用
# # DRONE_IP = "192.168.11.18" #実機用2
SKYCTRL_IP = "192.168.53.1"


##################yokomatsu
from kl_evaluation.msg import kl_eval, angular, kl_suzuki, kl_suzu
pai = 3.141592
kl_s = kl_eval()
feed = Twist()
semi_opt = kl_suzu()
# kl_flag = 0
import random
##############################################
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
		self.pub_inputR = rospy.Publisher('/controlinput', Int8, queue_size=1)
		self.pub_rpyt = rospy.Publisher('/anafi/cmd_rpyt', PilotingCommand, queue_size=1)
		

		rospy.Subscriber("/anafi/takeoff", Empty, self.takeoff_callback)
		rospy.Subscriber("/anafi/land", Empty, self.land_callback)
		rospy.Subscriber("/anafi/emergency", Empty, self.emergency_callback)
		rospy.Subscriber("/anafi/offboard", Bool, self.offboard_callback)
		rospy.Subscriber("/anafi/cmd_rpyt", PilotingCommand, self.rpyt_callback)
		rospy.Subscriber("/anafi/cmd_camera", CameraCommand, self.camera_callback)

		############################################
		rospy.Subscriber("/kl_msg", kl_eval, self.klCallback)
		rospy.Subscriber("/semi_opt", kl_suzu, self.semi_klCallback)
		rospy.Subscriber("/angular_msg", angular, self.angularCallback)
		self.kl_pub2 = rospy.Publisher("/kl_suzuki", kl_suzuki, queue_size=1)
		##############################################
		
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
		msg_image = Image()
		msg_image = self.bridge.cv2_to_imgmsg(cv2frame, "bgr8")
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
	################################################# 2022/06/21 yokomatsu slam
	def orbSlamCallback(self, msg):
		global slam_x, slam_y, slam_z, slam_roll, slam_pitch, slam_yaw
		slam_x = msg.pose.position.x
		slam_y = msg.pose.position.y
		slam_z = msg.pose.position.z
		q0 = msg.pose.orientation.w 
		q1 = msg.pose.orientation.x
		q2 = msg.pose.orientation.y
		q3 = msg.pose.orientation.z

		slam_roll = math.atan2(2*(q2*q3+q0*q1),(-1+2*(q0*q0+q3*q3)))  # rad
		roll = slam_roll*180/pai                                   # deg
        # cout<<"roll: "<<slam_roll<<endl;
		
		slam_pitch = math.asin(2*(q0*q2-q1*q3))                  # rad
		pitch = slam_pitch*180/pai                               # deg
		
		slam_yaw = math.atan2(2*(q1*q2+q0*q3),(-1+2*(q0*q0+q1*q1)))
		# rospy.loginfo("22222222")
		return slam_x, slam_y, slam_z, slam_roll, slam_pitch, slam_yaw
	#################################################
	################################################# 2022/09/13 yokomatsu
	def klCallback(self, msg):
		klkl = msg.kl
		if klkl > 0:
			kl_s.kl = msg.kl
		return kl_s
	
	def semi_klCallback(self, msg):
		ls = msg.kl_score
		if ls > 0:
			semi_opt.kl_score = msg.kl_score
			semi_opt.kl_x = msg.kl_x
			semi_opt.kl_y = msg.kl_y
			semi_opt.kl_z = msg.kl_z
			semi_opt.saiteki_kl = msg.saiteki_kl
			semi_opt.sinrai_kl = msg.sinrai_kl
			# pso_flag = 1
			# flag_gau = 1
		# return semi_opt, pso_flag, flag_gau
		return semi_opt
	
	def angularCallback(self, msg):
		feed.angular.z = msg.angular.z
		return feed.angular.z	

	print('kl_s', kl_s)
	print('feed.angular.z', feed.angular.z)
	kl_flag = 0
	#################################################

	def inputRcallback(self, msg):
		# self.inR = msg.inroll
		# self.inP = msg.inpitch
		rospy.loginfo("input_roll:%d", self.msg_input_roll)
		# rospy.loginfo("input_pitch:%d", self.msg_inputP)
		
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
############################################################# 2022/08/01 niwa ##########	
	# def sigmoid_drone(x):
	# 	return -10 / (1 + np.exp(-0.5 * x)) + 5.5
########################################################################################
	def run(self):
		global freq
		global flag1
		global move_flag
		global pub_go
		global num1
		scores = kl_suzuki()
		freq = 100       # [Hz]
		rate = rospy.Rate(freq) 
		
		rospy.logdebug('MaxTilt = %f [%f, %f]', self.drone.get_state(MaxTiltChanged)["current"], self.drone.get_state(MaxTiltChanged)["min"], self.drone.get_state(MaxTiltChanged)["max"])
		rospy.logdebug('MaxVerticalSpeed = %f [%f, %f]', self.drone.get_state(MaxVerticalSpeedChanged)["current"], self.drone.get_state(MaxVerticalSpeedChanged)["min"], self.drone.get_state(MaxVerticalSpeedChanged)["max"])
		rospy.logdebug('MaxRotationSpeed = %f [%f, %f]', self.drone.get_state(MaxRotationSpeedChanged)["current"], self.drone.get_state(MaxRotationSpeedChanged)["min"], self.drone.get_state(MaxRotationSpeedChanged)["max"])
		rospy.logdebug('MaxPitchRollRotationSpeed = %f [%f, %f]', self.drone.get_state(MaxPitchRollRotationSpeedChanged)["current"], self.drone.get_state(MaxPitchRollRotationSpeedChanged)["min"], self.drone.get_state(MaxPitchRollRotationSpeedChanged)["max"])
		rospy.logdebug('MaxDistance = %f [%f, %f]', self.drone.get_state(MaxDistanceChanged)["current"], self.drone.get_state(MaxDistanceChanged)["min"], self.drone.get_state(MaxDistanceChanged)["max"])
		rospy.logdebug('MaxAltitude = %f [%f, %f]', self.drone.get_state(MaxAltitudeChanged)["current"], self.drone.get_state(MaxAltitudeChanged)["min"], self.drone.get_state(MaxAltitudeChanged)["max"])
		rospy.logdebug('NoFlyOverMaxDistance = %i', self.drone.get_state(NoFlyOverMaxDistanceChanged)["shouldNotFlyOver"])
		rospy.logdebug('BankedTurn = %i', self.drone.get_state(BankedTurnChanged)["state"])
		maxTiltAction = self.drone(MaxTilt(10, _timeout=1)).wait()
	
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
			
			def btn_up():
				global flag1, way_x, way_y, way_z, error_x, error_y, error_z, diff_go_x, diff_go_y, Kppx, Kppy, Kppz, Kdpx, Kdpy, Kdpz, integ_x, integ_y
				way_x = 0.0      #目標値
				way_y = 0.0      #
				way_z = 1.5      #

				error_x = 0
				error_y = 0
				error_z = 0

				diff_go_x = 0
				diff_go_y = 0

				# diff_px = 0
				# diff_py = 0
				# diff_pz = 0

				# error_x_prev = 0
				# error_y_prev = 0
				# error_z_prev = 0

				integ_x = 0
				integ_y = 0

				#Kppx = 30   # x # 比例ゲイン
				#Kppy = 30   # y
				Kppx = 10
				Kppy = 15
				Kppz = 1.0  

				#Kdpx = 100  # x # 微分ゲイン
				#Kdpy = 90 # y
				Kdpx = 60
				Kdpy = 60
				Kdpz = 0.1

				# Kppx = 3  # 比例ゲイン
				# Kppy = 1.8
				# Kppz = 1.0

				# Kdpx = 5  # 微分ゲイン
				# Kdpy = 4
				# Kdpz = 0.1

				Kipx = 10   # 積分ゲイン
				Kipy = 0
				Kipz = 1.0

				# inR = 0
				# inP = 0

				maxRP = 3   # 設定した最大角度
######################################## 2022/09/28 yokomatsu　自律撮影用変数設定
				move_flag = 0
				num = 0
				yoko = 1.5
				tate = 1.0
				takasa = 1.8
				hikusa = 0.8
				num1 = 0
				best_score = 100
				time_rec = rospy.Time.now()
				time0 = time.time()
#########################################
###################################### 2022/08/02 niwa #####################################
				def x_sigmoid(x):
					return 1 / (1 + np.exp(-x * 1.1)) * 60 - 30
				
				def y_sigmoid(x):
					return 1 / (1 + np.exp(-x * 1.1)) * 60 - 30
				
				def z_sigmoid(x):
					return 1 / (1 + np.exp(-x * 7)) * 60 - 30

				def angular_sigmoid(x):
					return 1 / (1 + np.exp(-x * 8)) * 200 - 100

				def sigmoid(x):    # 観測更新用のシグモイド，今回まだ使うかわからんからパラメータの調整もしていない
					return 1.0 / (1.0 + np.exp(- (x - 10) * 0.5)) / 2

				def control1(dif, a):
					global gogo_x, gogo_y, diff_go_x, diff_go_y
					go_x = dif*np.cos(a)
					go_y = dif*np.sin(a)
					integ_x =+ go_x
					integ_y =+ go_y 
					gogo1_x = Kppx * go_x + Kdpx * (go_x - diff_go_x) + Kipx * integ_x
					gogo1_y = Kppy * go_y + Kdpy * (go_y - diff_go_y) + Kipy + integ_y
					gogo_x = int(gogo1_x)
					gogo_y = int(gogo1_y)
					# print('11111111111111111111111')
					diff_go_x = go_x
					diff_go_y = go_y
					return gogo_x, gogo_y

				def control2(dif, a):
					global gogo_x, gogo_y, diff_go_x, diff_go_y
					go_x = dif*np.cos(a)
					go_y = dif*np.sin(a)
					integ_x =+ go_x
					integ_y =+ go_y 
					gogo1_x = Kppx * go_x + Kdpx * (go_x - diff_go_x) + Kipx * integ_x
					gogo1_y = Kppy * go_y + Kdpy * (go_y - diff_go_y) + Kipy + integ_y
					gogo_x = int(gogo1_x)
					gogo_y = int(gogo1_y)
					# print('22222222222222222222222222222')
					diff_go_x = go_x
					diff_go_y = go_y
					return gogo_x, gogo_y

				def control3(dif, a):
					global gogo_x, gogo_y, diff_go_x, diff_go_y
					go_x = - dif*np.cos(a)
					go_y = - dif*np.sin(a)
					integ_x =+ go_x
					integ_y =+ go_y 
					gogo1_x = Kppx * go_x + Kdpx * (go_x - diff_go_x) + Kipx * integ_x
					gogo1_y = Kppy * go_y + Kdpy * (go_y - diff_go_y) + Kipy + integ_y
					gogo_x = int(gogo1_x)
					gogo_y = int(gogo1_y)
					# print('33333333333333333333333333333')
					diff_go_x = go_x
					diff_go_y = go_y
					return gogo_x, gogo_y

				def control4(dif, a):
					global gogo_x, gogo_y, diff_go_x, diff_go_y
					go_x = - dif*np.cos(a)
					go_y = - dif*np.sin(a)
					integ_x =+ go_x
					integ_y =+ go_y 
					gogo1_x = Kppx * go_x + Kdpx * (go_x - diff_go_x) + Kipx * integ_x
					gogo1_y = Kppy * go_y + Kdpy * (go_y - diff_go_y) + Kipy + integ_y
					gogo_x = int(gogo1_x)
					gogo_y = int(gogo1_y)
					# print('444444444444444444444444')
					diff_go_x = go_x
					diff_go_y = go_y
					return gogo_x, gogo_y

				while flag1 == 2:
					global a, dif
					time_rec = rospy.Time.now()
					with open('/home/dars/デスクトップ/slam.csv', 'a', newline='') as f:
						writer = csv.writer(f)
						data = [slam_x, slam_y, slam_z, slam_roll, kl_s.kl, time_rec]
						writer.writerow(data)
					rospy.loginfo("best_score: %f", best_score)
					rospy.loginfo('move_flag: %d', move_flag)
					if move_flag == 0:
						with open('/home/dars/デスクトップ/data.csv', 'a', newline='') as f:
							writer = csv.writer(f)
							data = [slam_x, slam_y, slam_z, slam_roll, kl_s.kl, time_rec]
							writer.writerow(data)
						now_score = float(kl_s.kl)
						now_x = float(slam_x)
						now_y = float(slam_y)
						now_z = float(slam_z)
						num = num + 1
						score = kl_suzu()
						score.num = num
						score.kl_score = float(kl_s.kl)
						score.kl_x = float(slam_x)
						score.kl_y = float(slam_y)
						score.kl_z = float(slam_z)
						scores.KL.append(score)
						self.kl_pub2.publish(scores)
						ran_x = random.randint(0, 100)
						ran_y = random.randint(0, 100)
						ran_z = random.randint(0, 100)
						way_x = - (yoko/2.0) + (yoko/3.0) * ran_x/100
						way_y = ran_y*tate/100 - (tate/2)
						way_z = 0.8 + ran_z/100
						if way_x > yoko/2:
							way_x = yoko/2
						if way_x < -yoko/2:
							way_x = -yoko/2
						if way_y > tate/2:
							way_y = tate/2
						if way_y < -tate/2:
							way_y = -tate/2
						if way_z > takasa:
							way_z = takasa
						if way_z < hikusa:
							way_z = hikusa
						time1 = time.time()
						move_flag = 1
##############################################################################
					if move_flag == 1:
						time2 = time.time() - time1
						error_x = way_x - slam_x      # 偏差(P用)
						error_y = way_y - slam_y
						error_z = way_z - slam_z
						z_control = z_sigmoid(error_z)
						zz_control = int(z_control)
						angular_control = angular_sigmoid(feed.angular.z)
						aangular_control = int(angular_control)
						distance = np.sqrt(np.square(error_x) + np.square(error_y) + np.square(error_z))
						dif = np.sqrt(np.square(error_x) + np.square(error_y))
						target_theta = np.arctan(error_y/error_x)
						a = target_theta - slam_yaw
						if error_x > 0 and error_y > 0:
							control1(dif, a)
						if error_x > 0 and error_y < 0:
							control2(dif, a)
						if error_x < 0 and error_y > 0:
							control3(dif, a)
						if error_x < 0 and error_y < 0:
							control4(dif, a)
						if feed.angular.z > 0 :
							# rospy.loginfo('right')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									aangular_control,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if feed.angular.z < 0 :
							# rospy.loginfo('left')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									aangular_control,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if feed.angular.z == 0 :
							# rospy.loginfo('none')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									0,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if distance < 0.2:
							with open('/home/dars/デスクトップ/data.csv', 'a', newline='') as f:
								writer = csv.writer(f)
								data = [slam_x, slam_y, slam_z, slam_roll, kl_s.kl, time_rec]
								writer.writerow(data)
							now_score = float(kl_s.kl)
							now_x = float(slam_x)
							now_y = float(slam_y)
							now_z = float(slam_z)
							num = num + 1
							score = kl_suzu()
							score.num = num
							score.kl_score = float(kl_s.kl)
							score.kl_x = float(slam_x)
							score.kl_y = float(slam_y)
							score.kl_z = float(slam_z)
							scores.KL.append(score)
							self.kl_pub2.publish(scores)
							rospy.loginfo("num: %d", num)
							if num1 == 0:
								move_flag = 0
								num1 = 1
							else:
								move_flag = 3
								num1 = 0
						if int(time2) > 5:
							with open('/home/dars/デスクトップ/data.csv', 'a', newline='') as f:
								writer = csv.writer(f)
								data = [slam_x, slam_y, slam_z, slam_roll, kl_s.kl, time_rec]
								writer.writerow(data)
							now_score = float(kl_s.kl)
							now_x = float(slam_x)
							now_y = float(slam_y)
							now_z = float(slam_z)
							num = num + 1
							score = kl_suzu()
							score.num = num
							score.kl_score = float(kl_s.kl)
							score.kl_x = float(slam_x)
							score.kl_y = float(slam_y)
							score.kl_z = float(slam_z)
							scores.KL.append(score)
							self.kl_pub2.publish(scores)
							move_flag = 0
####################################################################################
					if move_flag == 3:
						ran_x = random.randint(0, 100)
						ran_y = random.randint(0, 100)
						ran_z = random.randint(0, 100)
						way_x = - (yoko/6.0) + (yoko/3.0) * ran_x/100
						way_y = ran_y*tate/100 - (tate/2)
						way_z = 0.8 + ran_z/100
						if way_x > yoko/2:
							way_x = yoko/2
						if way_x < -yoko/2:
							way_x = -yoko/2
						if way_y > tate/2:
							way_y = tate/2
						if way_y < -tate/2:
							way_y = -tate/2
						if way_z > takasa:
							way_z = takasa
						if way_z < hikusa:
							way_z = hikusa
						time1 = time.time()
						move_flag = 4
					if move_flag == 4:
						time2 = time.time() - time1
						error_x = way_x - slam_x      # 偏差(P用)
						error_y = way_y - slam_y
						error_z = way_z - slam_z
						z_control = z_sigmoid(error_z)
						zz_control = int(z_control)
						angular_control = angular_sigmoid(feed.angular.z)
						aangular_control = int(angular_control)
						distance = np.sqrt(np.square(error_x) + np.square(error_y) + np.square(error_z))
						dif = np.sqrt(np.square(error_x) + np.square(error_y))
						target_theta = np.arctan(error_y/error_x)
						a = target_theta - slam_yaw
						if error_x > 0 and error_y > 0:
							control1(dif, a)
						if error_x > 0 and error_y < 0:
							control2(dif, a)
						if error_x < 0 and error_y > 0:
							control3(dif, a)
						if error_x < 0 and error_y < 0:
							control4(dif, a)
						if feed.angular.z > 0 :
							# rospy.loginfo('right')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									aangular_control,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if feed.angular.z < 0 :
							# rospy.loginfo('left')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									aangular_control,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if feed.angular.z == 0 :
							# rospy.loginfo('none')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									0,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if distance < 0.2:
							with open('/home/dars/デスクトップ/data.csv', 'a', newline='') as f:
								writer = csv.writer(f)
								data = [slam_x, slam_y, slam_z, slam_roll, kl_s.kl, time_rec]
								writer.writerow(data)
							now_score = float(kl_s.kl)
							now_x = float(slam_x)
							now_y = float(slam_y)
							now_z = float(slam_z)
							num = num + 1
							score = kl_suzu()
							score.num = num
							score.kl_score = float(kl_s.kl)
							score.kl_x = float(slam_x)
							score.kl_y = float(slam_y)
							score.kl_z = float(slam_z)
							scores.KL.append(score)
							self.kl_pub2.publish(scores)
							rospy.loginfo("num: %d", num)
							if num1 == 0:
								move_flag = 3
								num1 = 1
							else:
								move_flag = 5
								num1 = 0
						if int(time2) > 5:
							with open('/home/dars/デスクトップ/data.csv', 'a', newline='') as f:
								writer = csv.writer(f)
								data = [slam_x, slam_y, slam_z, slam_roll, kl_s.kl, time_rec]
								writer.writerow(data)
							now_score = float(kl_s.kl)
							now_x = float(slam_x)
							now_y = float(slam_y)
							now_z = float(slam_z)
							num = num + 1
							score = kl_suzu()
							score.num = num
							score.kl_score = float(kl_s.kl)
							score.kl_x = float(slam_x)
							score.kl_y = float(slam_y)
							score.kl_z = float(slam_z)
							scores.KL.append(score)
							self.kl_pub2.publish(scores)
							move_flag = 3
##########################################################################
					if move_flag == 5:
						ran_x = random.randint(0, 100)
						ran_y = random.randint(0, 100)
						ran_z = random.randint(0, 100)
						way_x = (yoko/6.0) + (yoko/3.0) * ran_x/100
						way_y = ran_y*tate/100 - (tate/2)
						way_z = 0.8 + ran_z/100
						if way_x > yoko/2:
							way_x = yoko/2
						if way_x < -yoko/2:
							way_x = -yoko/2
						if way_y > tate/2:
							way_y = tate/2
						if way_y < -tate/2:
							way_y = -tate/2
						if way_z > takasa:
							way_z = takasa
						if way_z < hikusa:
							way_z = hikusa
						time1 = time.time()
						move_flag = 6
					if move_flag == 6:
						time2 = time.time() - time1
						error_x = way_x - slam_x      # 偏差(P用)
						error_y = way_y - slam_y
						error_z = way_z - slam_z
						z_control = z_sigmoid(error_z)
						zz_control = int(z_control)
						angular_control = angular_sigmoid(feed.angular.z)
						aangular_control = int(angular_control)
						distance = np.sqrt(np.square(error_x) + np.square(error_y) + np.square(error_z))
						dif = np.sqrt(np.square(error_x) + np.square(error_y))
						target_theta = np.arctan(error_y/error_x)
						a = target_theta - slam_yaw
						if error_x > 0 and error_y > 0:
							control1(dif, a)
						if error_x > 0 and error_y < 0:
							control2(dif, a)
						if error_x < 0 and error_y > 0:
							control3(dif, a)
						if error_x < 0 and error_y < 0:
							control4(dif, a)
						if feed.angular.z > 0 :
							# rospy.loginfo('right')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									aangular_control,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if feed.angular.z < 0 :
							# rospy.loginfo('left')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									aangular_control,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if feed.angular.z == 0 :
							# rospy.loginfo('none')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									0,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if distance < 0.2:
							with open('/home/dars/デスクトップ/data.csv', 'a', newline='') as f:
								writer = csv.writer(f)
								data = [slam_x, slam_y, slam_z, slam_roll, kl_s.kl, time_rec]
								writer.writerow(data)
							now_score = float(kl_s.kl)
							now_x = float(slam_x)
							now_y = float(slam_y)
							now_z = float(slam_z)
							num = num + 1
							score = kl_suzu()
							score.num = num
							score.kl_score = float(kl_s.kl)
							score.kl_x = float(slam_x)
							score.kl_y = float(slam_y)
							score.kl_z = float(slam_z)
							scores.KL.append(score)
							self.kl_pub2.publish(scores)
							rospy.loginfo("num: %d", num)
							if num1 == 0:
								move_flag = 5
								num1 = 1
							else:
								move_flag = 7
								num1 = 0
						if int(time2) > 5:
							with open('/home/dars/デスクトップ/data.csv', 'a', newline='') as f:
								writer = csv.writer(f)
								data = [slam_x, slam_y, slam_z, slam_roll, kl_s.kl, time_rec]
								writer.writerow(data)
							now_score = float(kl_s.kl)
							now_x = float(slam_x)
							now_y = float(slam_y)
							now_z = float(slam_z)
							num = num + 1
							score = kl_suzu()
							score.num = num
							score.kl_score = float(kl_s.kl)
							score.kl_x = float(slam_x)
							score.kl_y = float(slam_y)
							score.kl_z = float(slam_z)
							scores.KL.append(score)
							self.kl_pub2.publish(scores)
							move_flag = 5
########################################################################################################################
					if move_flag == 7:
						rospy.loginfo('semi_opt.kl_score: %f', semi_opt.kl_score)
						way_x = semi_opt.kl_x + sigmoid(semi_opt.kl_score) * random.random()
						way_y = semi_opt.kl_y + sigmoid(semi_opt.kl_score) * random.random()
						way_z = semi_opt.kl_z + sigmoid(semi_opt.kl_score) * random.random()
						if way_x > yoko/2:
							way_x = yoko/2
						if way_x < -yoko/2:
							way_x = -yoko/2
						if way_y > tate/2:
							way_y = tate/2
						if way_y < -tate/2:
							way_y = -tate/2
						if way_z > takasa:
							way_z = takasa
						if way_z < hikusa:
							way_z = hikusa
						time1 = time.time()
						move_flag = 8
					if move_flag == 8:
						time2 = time.time() - time1
						error_x = way_x - slam_x      # 偏差(P用)
						error_y = way_y - slam_y
						error_z = way_z - slam_z
						z_control = z_sigmoid(error_z)
						zz_control = int(z_control)
						angular_control = angular_sigmoid(feed.angular.z)
						aangular_control = int(angular_control)
						distance = np.sqrt(np.square(error_x) + np.square(error_y) + np.square(error_z))
						dif = np.sqrt(np.square(error_x) + np.square(error_y))
						target_theta = np.arctan(error_y/error_x)
						a = target_theta - slam_yaw
						if error_x > 0 and error_y > 0:
							control1(dif, a)
						if error_x > 0 and error_y < 0:
							control2(dif, a)
						if error_x < 0 and error_y > 0:
							control3(dif, a)
						if error_x < 0 and error_y < 0:
							control4(dif, a)
						if feed.angular.z > 0 :
							# rospy.loginfo('right')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									aangular_control,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if feed.angular.z < 0 :
							# rospy.loginfo('left')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									aangular_control,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if feed.angular.z == 0 :
							# rospy.loginfo('none')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									0,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if distance < 0.2:
							with open('/home/dars/デスクトップ/data.csv', 'a', newline='') as f:
								writer = csv.writer(f)
								data = [slam_x, slam_y, slam_z, slam_roll, kl_s.kl, time_rec]
								writer.writerow(data)
							now_score = float(kl_s.kl)
							now_x = float(slam_x)
							now_y = float(slam_y)
							now_z = float(slam_z)
							num = num + 1
							score = kl_suzu()
							score.num = num
							score.kl_score = float(kl_s.kl)
							score.kl_x = float(slam_x)
							score.kl_y = float(slam_y)
							score.kl_z = float(slam_z)
							scores.KL.append(score)
							self.kl_pub2.publish(scores)
							rospy.loginfo("num: %d", num)
							move_flag = 7
						if int(time2) > 5:
							with open('/home/dars/デスクトップ/data.csv', 'a', newline='') as f:
								writer = csv.writer(f)
								data = [slam_x, slam_y, slam_z, slam_roll, kl_s.kl, time_rec]
								writer.writerow(data)
							now_score = float(kl_s.kl)
							now_x = float(slam_x)
							now_y = float(slam_y)
							now_z = float(slam_z)
							num = num + 1
							score = kl_suzu()
							score.num = num
							score.kl_score = float(kl_s.kl)
							score.kl_x = float(slam_x)
							score.kl_y = float(slam_y)
							score.kl_z = float(slam_z)
							scores.KL.append(score)
							self.kl_pub2.publish(scores)
							move_flag = 7
######################################################################							
					if best_score > now_score:
						best_score = now_score
						best_x = now_x
						best_y = now_y
						best_z = now_z
########################################################################
					if time.time() - time0 > 120:
						way_x = best_x
						way_y = best_y
						way_z = best_z
						if way_x > yoko/2:
							way_x = yoko/2
						if way_x < -yoko/2:
							way_x = -yoko/2
						if way_y > tate/2:
							way_y = tate/2
						if way_y < -tate/2:
							way_y = -tate/2
						if way_z > takasa:
							way_z = takasa
						if way_z < hikusa:
							way_z = hikusa
						time1 = time.time()
						move_flag = 1000
					if move_flag == 1000:
						time2 = time.time() - time1
						error_x = way_x - slam_x      # 偏差(P用)
						error_y = way_y - slam_y
						error_z = way_z - slam_z
						z_control = z_sigmoid(error_z)
						zz_control = int(z_control)
						angular_control = angular_sigmoid(feed.angular.z)
						aangular_control = int(angular_control)
						distance = np.sqrt(np.square(error_x) + np.square(error_y) + np.square(error_z))
						dif = np.sqrt(np.square(error_x) + np.square(error_y))
						target_theta = np.arctan(error_y/error_x)
						a = target_theta - slam_yaw
						if error_x > 0 and error_y > 0:
							control1(dif, a)
						if error_x > 0 and error_y < 0:
							control2(dif, a)
						if error_x < 0 and error_y > 0:
							control3(dif, a)
						if error_x < 0 and error_y < 0:
							control4(dif, a)
						if feed.angular.z > 0 :
							# rospy.loginfo('right')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									aangular_control,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if feed.angular.z < 0 :
							# rospy.loginfo('left')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									aangular_control,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if feed.angular.z == 0 :
							# rospy.loginfo('none')
							self.drone(
								PCMD(
									1,
									gogo_x,
									gogo_y,
									0,
									zz_control,
									timestampAndSeqNum=0,
								)
							)
							time.sleep(0.5)
						if distance < 0.2:
							with open('/home/dars/デスクトップ/data.csv', 'a', newline='') as f:
								writer = csv.writer(f)
								data = [slam_x, slam_y, slam_z, slam_roll, kl_s.kl, time_rec]
								writer.writerow(data)
							move_flag = 4649
						if int(time2) > 5:
							with open('/home/dars/デスクトップ/data.csv', 'a', newline='') as f:
								writer = csv.writer(f)
								data = [slam_x, slam_y, slam_z, slam_roll, kl_s.kl, time_rec]
								writer.writerow(data)
							move_flag = 4649
########################################################################
					if move_flag == 4649:
						self.drone(Landing())
############################################################################################################
			connection = self.drone.connection_state()
			rospy.loginfo("111");
			root = tk.Tk()
			rospy.loginfo("12345");
			root.title(u"keyboard controll")
			def key_event(e):
				global flag1
				key = e.keysym
				print("入力したキー:", e.keysym)
				if key == "t":
					self.drone(TakeOff())
				if key == "l":
					self.drone(
						PCMD(
							1,
							0,
							0,
							0,
							0,
							timestampAndSeqNum=0							
						)
					)
					self.drone(Landing())
				#ポイント移動X軸前
				if key == "w":
					self.drone(PCMD(
						1,
						0,
						60,
						0,
						0,
						timestampAndSeqNum=0,
					))
				if key == "s":
					self.drone(PCMD(
						1,
						0,
						-60,
						0,
						0,
						timestampAndSeqNum=0,
					))
				#ポイント移動Y軸
				if key == "a":
					self.drone(PCMD(
						1,
						-60,
						0,
						0,
						0,
						timestampAndSeqNum=0,
					))
				if key == "d":
					self.drone(PCMD(
						1,
						60,
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
						100,
						0,
						timestampAndSeqNum=0,
					))
				if key == "q":
					self.drone(PCMD(
						1,
						0,
						0,
						-100,
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
						-30,
						timestampAndSeqNum=0,
					))
				if key == "Up":
					self.drone(PCMD(
						1,
						0,
						0,
						0,
						30,
						timestampAndSeqNum=0,
					))
				if key == "g":  #カメラ角度移動（水平）
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
				if key == "h":   #下向き
					self.drone(gimbal.set_target( # https://developer.parrot.com/docs/olympe/arsdkng_gimbal.html#olympe.messages.gimbal.set_target
						gimbal_id=0,
						control_mode='position', # {'position', 'velocity'}
						yaw_frame_of_reference='none',
						yaw=0.0,
						pitch_frame_of_reference='absolute', # {'absolute', 'relative', 'none'}
						pitch=-20,
						roll_frame_of_reference=self.gimbal_frame, # {'absolute', 'relative', 'none'}
						roll=0
						)
					)
################################################## 2022/06/23  slam座標における原点に移動
				if key == "7":
					threadB = threading.Thread(target=btn_up)
					threadB.start()
				if key == "8":
					flag1 = 10
					print("flag1:", flag1)
					self.drone(
						PCMD(
							1,
							0,
							0,
							0,
							0,
							timestampAndSeqNum=0,
						)
					)
					time.sleep(0.01)
				if key == "9":
					flag1 = 2
					print("flag1:", flag1)					
				
				if key == "m": 
					# self.drone(CancelMoveBy(_timeout = 10, _no_expect = False, _float_tol = (1e-07, 1e-09)))
					self.drone(
						PCMD(
							1,
							0,
							0,
							0,
							0,
							timestampAndSeqNum=0,
						)
					)
					time.sleep(0.01)
#######################################################
			root.bind("<KeyPress>", key_event)
			root.mainloop()
#######################################################
			if getattr(connection, 'OK') == False:
				rospy.logfatal(getattr(connection, 'message'))
				rospy.loginfo("222");
				self.disconnect()
				self.connect()


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

if __name__ == '__main__':
	rospy.init_node('anafi_bridge', anonymous = False)
	rospy.loginfo("AnafiBridge is running...")
	anafi = Anafi()
	########### 2022/06/21 yokomatsu slam subscribe
	sub_slam = rospy.Subscriber("/bebop/pose", PoseStamped, anafi.orbSlamCallback)
	# sub_input = rospy.Subscriber("/controlinput", Int8, anafi.inputRcallback)
	sub_rpyt = rospy.Subscriber("/anafi/cmd_rpyt", PilotingCommand, anafi.rpyt_callback)
	# sub_rpyt = rospy.Subscriber("/anafi/cmd_rpyt", PilotingCommand, anafi.nowrpyt_callback)
	sub_skycontroller = rospy.Subscriber("/skycontroller/command", SkyControllerCommand, anafi.switch_manual)


	###########	
	anafi.drone(gimbal.set_target( # https://developer.parrot.com/docs/olympe/arsdkng_gimbal.html#olympe.messages.gimbal.set_target
						gimbal_id=0,
						control_mode='position', # {'position', 'velocity'}
						yaw_frame_of_reference='none',
						yaw=0.0,
						pitch_frame_of_reference='absolute', # {'absolute', 'relative', 'none'}
						pitch=-20,
						roll_frame_of_reference=anafi.gimbal_frame, # {'absolute', 'relative', 'none'}
						roll=-40,
						)
					)
	try:
		anafi.run()
	except rospy.ROSInterruptException:
		traceback.print_exc()
		pass
