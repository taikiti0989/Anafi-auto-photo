#ifndef POSE_REPUBLISHER_H
#define POSE_REPUBLISHER_H

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <tf/tf.h>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/tracking.hpp"
#include <ar_track_alvar_msgs/AlvarMarkers.h>
#include <tf/transform_broadcaster.h>

#define PI    3.141592653589793238

class PoseRepublisher
{
public:
	///params
	bool update_transform;
	double scaling_factor;
	std::string orb_topic_name, ar_topic_name, true_pose_topic_name;
	////
    PoseRepublisher();
	~PoseRepublisher();
    ros::NodeHandle nh, private_nh;
	ros::Subscriber orb_topic_sub;
	ros::Subscriber ar_topic_sub;
	ros::Publisher true_pose_pub;
    bool init_orb;
    bool init_ar;
    bool init_true;
    bool update_ar;
	int count;
 
	void save();
	        std::string result_path;
	        std::ofstream TruePose_o;
	        std::ofstream ar_o;
	        std::ofstream T_o;
	        std::ofstream orb_o;
       void GetRPY(const geometry_msgs::Quaternion &q, double &roll,double &pitch,double &yaw);

	void orb_topicCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
			geometry_msgs::PoseStamped orb_topic;
			tf::Transform orb_pose;
       void orb_CalculateTransform();
	
    void ar_topicCallback(const ar_track_alvar_msgs::AlvarMarkers::ConstPtr& msg);
			geometry_msgs::PoseStamped ar_topic;
			geometry_msgs::PoseStamped ar_topicMsg;
			tf::Transform ar_pose;
       void ar_CalculateTransform();
    
       void CalculateT();
	        tf::Transform T;
	        geometry_msgs::PoseStamped T_msg;

		void CalculateTruePose();
		    tf::Transform tf_true_pose;

            tf::TransformBroadcaster tb;
			std::string tf_pose;

			geometry_msgs::PoseStamped TruePose;
};
#endif
