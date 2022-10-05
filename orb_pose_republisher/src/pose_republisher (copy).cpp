#include "pose_republisher.h"

int main(int argc, char **argv)
{
	ros::init(argc, argv, "pose_republisher");
	PoseRepublisher pr;

	ros::Rate rate(50);
	while(ros::ok())
	{
		ros::spinOnce();
		if (pr.init_ar && pr.init_orb && pr.init_true)
		{
			pr.CalculateTruePose();
			if (pr.init_true == true)
				pr.save();
		}	
		rate.sleep();
	}
	return 0;
}

PoseRepublisher::PoseRepublisher() :private_nh("~")
{
	if (!private_nh.getParam("update_transform", update_transform))
        update_transform = false;
	if (!private_nh.getParam("scaling_factor", scaling_factor))
        scaling_factor = 2.45;
	ROS_ERROR("scaling factor: %lf", scaling_factor);
	if (!private_nh.getParam("result_path", result_path))
        result_path = "/home/das-note-8/Desktop/Result/";
	if (!private_nh.getParam("orb_topic", orb_topic_name))
        orb_topic_name = "/orb_slam2_mono/pose";
	if (!private_nh.getParam("ar_topic", ar_topic_name))
        ar_topic_name = "/ar_pose_marker";
	if (!private_nh.getParam("pose_topic", true_pose_topic_name))
        true_pose_topic_name = "bebop/true_pose";

	orb_topic_sub = nh.subscribe(orb_topic_name, 1, &PoseRepublisher::orb_topicCallback, this);
    ar_topic_sub = nh.subscribe(ar_topic_name, 1, &PoseRepublisher::ar_topicCallback, this);
	true_pose_pub = nh.advertise<geometry_msgs::PoseStamped>(true_pose_topic_name, 1);

	init_ar = false;
	init_orb = false;
	init_true = false;
	update_ar = false;
	count = 0;

	TruePose_o = std::ofstream(std::string(result_path + "TruePose.csv").c_str());
  	TruePose_o << "count, tx, ty, tz, qx, qy, qz, qw, roll, pitch, yaw" <<std::endl;
	ar_o = std::ofstream(std::string(result_path + "ar.csv").c_str());
  	ar_o << "count, tx, ty, tz, qx, qy, qz, qw, roll, pitch, yaw" <<std::endl;
  	T_o = std::ofstream(std::string(result_path + "T_Pose.csv").c_str());
  	T_o << "count, tx, ty, tz, qx, qy, qz, qw, roll, pitch, yaw" <<std::endl;
  	orb_o = std::ofstream(std::string(result_path + "ORBPose.csv").c_str());
  	orb_o << "count, tx, ty, tz, qx, qy, qz, qw, roll, pitch, yaw" <<std::endl;
}
PoseRepublisher::~PoseRepublisher()
{}

void PoseRepublisher::save()
{
	if (init_true && init_ar)
	{
		double roll,pitch,yaw;
	    GetRPY(TruePose.pose.orientation, roll,pitch,yaw);
		TruePose_o << count <<","<< TruePose.pose.position.x <<","<< TruePose.pose.position.y <<","<< TruePose.pose.position.z <<","<< TruePose.pose.orientation.x <<","<< TruePose.pose.orientation.y <<","<< TruePose.pose.orientation.z <<","<< TruePose.pose.orientation.w <<","<< roll <<","<< pitch <<","<< yaw <<std::endl; 
		double roll_,pitch_,yaw_;
	    GetRPY(ar_topicMsg.pose.orientation, roll_,pitch_,yaw_);
		ar_o << count <<","<< ar_topicMsg.pose.position.x <<","<< ar_topicMsg.pose.position.y <<","<< ar_topicMsg.pose.position.z <<","<< ar_topicMsg.pose.orientation.x <<","<< ar_topicMsg.pose.orientation.y <<","<< ar_topicMsg.pose.orientation.z <<","<< ar_topicMsg.pose.orientation.w<<","<< roll_ <<","<< pitch_ <<","<< yaw_ <<std::endl; 
		double roll_T,pitch_T,yaw_T;
	    GetRPY(T_msg.pose.orientation, roll_T,pitch_T,yaw_T);
		T_o << count <<","<< T_msg.pose.position.x <<","<< T_msg.pose.position.y <<","<< T_msg.pose.position.z <<","<< T_msg.pose.orientation.x <<","<< T_msg.pose.orientation.y <<","<< T_msg.pose.orientation.z <<","<< T_msg.pose.orientation.w<<","<< roll_T <<","<< pitch_T <<","<< yaw_T<<std::endl;
		double s_roll,s_pitch,s_yaw;
		GetRPY(orb_topic.pose.orientation, s_roll,s_pitch,s_yaw);
		double s_roll_,s_pitch_,s_yaw_;
		geometry_msgs::Pose p;
		tf::poseTFToMsg(orb_pose,p);
		GetRPY(p.orientation, s_roll_,s_pitch_,s_yaw_);
		orb_o << count <<","<< orb_topic.pose.position.x <<","<< orb_topic.pose.position.y <<","<< orb_topic.pose.position.z <<","<< orb_topic.pose.orientation.x <<","<< orb_topic.pose.orientation.y <<","<< orb_topic.pose.orientation.z <<","<< orb_topic.pose.orientation.w<<","<< s_roll <<","<< s_pitch <<","<< s_yaw
		<<","<< p.position.x <<","<< p.position.y <<","<< p.position.z <<","<< p.orientation.x <<","<< p.orientation.y <<","<< p.orientation.z  <<","<< p.orientation.w  <<","<< s_roll_  <<","<< s_pitch_  <<","<< s_yaw_ <<std::endl;
    }
}

void PoseRepublisher::GetRPY(const geometry_msgs::Quaternion &q, double &roll,double &pitch,double &yaw)
{
  tf::Quaternion quat(q.x,q.y,q.z,q.w);
  tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
}

void PoseRepublisher::orb_topicCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
	init_orb = true;
	orb_topic = *msg;
	if (!std::isnan(orb_topic.pose.position.x) && !std::isnan(orb_topic.pose.position.y) && !std::isnan(orb_topic.pose.position.z)
		&& !std::isnan(orb_topic.pose.orientation.x) && !std::isnan(orb_topic.pose.orientation.y) && !std::isnan(orb_topic.pose.orientation.z) && !std::isnan(orb_topic.pose.orientation.w))
	{
		orb_CalculateTransform();

		if (init_ar && init_orb && update_ar)
		{
			if (update_transform || !init_true)
			{
				CalculateT();
			}
		}
			
	}
	else
		ROS_INFO("nan");
}

void PoseRepublisher::orb_CalculateTransform()
{
    tf::Quaternion orb_rotation (orb_topic.pose.orientation.x, orb_topic.pose.orientation.y, orb_topic.pose.orientation.z, orb_topic.pose.orientation.w);
	tf::Vector3 orb_position (orb_topic.pose.position.x*scaling_factor, orb_topic.pose.position.y*scaling_factor, orb_topic.pose.position.z*scaling_factor);
	orb_pose = tf::Transform(orb_rotation, orb_position);

     tf::Vector3 p(0,0,0);
     tf::Quaternion q;
     q.setRPY(PI/2,-PI/2,0); 
	 tf::Transform t(q,p);
	 orb_pose = t * orb_pose;
	 orb_pose = orb_pose * t.inverse();
}

void PoseRepublisher::ar_topicCallback(const ar_track_alvar_msgs::AlvarMarkers::ConstPtr& msg)
{
	ar_track_alvar_msgs::AlvarMarkers markers = *msg;
	if (markers.markers.size()>0 && init_orb)
	{
		if (!std::isnan(markers.markers[0].pose.pose.position.x) && !std::isnan(markers.markers[0].pose.pose.position.y) && !std::isnan(markers.markers[0].pose.pose.position.z)
		&& !std::isnan(markers.markers[0].pose.pose.orientation.x) && !std::isnan(markers.markers[0].pose.pose.orientation.y) && !std::isnan(markers.markers[0].pose.pose.orientation.z) && !std::isnan(markers.markers[0].pose.pose.orientation.w))
		{
			ar_topic = markers.markers[0].pose;
			update_ar = true;
			init_ar = true;
			ar_CalculateTransform();
		}
	}
}

void PoseRepublisher::ar_CalculateTransform()
{
	tf::Quaternion ar_rotation (ar_topic.pose.orientation.x, ar_topic.pose.orientation.y, ar_topic.pose.orientation.z, ar_topic.pose.orientation.w);
	tf::Vector3 ar_position (ar_topic.pose.position.x, ar_topic.pose.position.y, ar_topic.pose.position.z);
	tf::Transform marker_pose(ar_rotation, ar_position);
	ar_pose = marker_pose.inverse();
	tf::poseTFToMsg(ar_pose,ar_topicMsg.pose);
	tb.sendTransform(tf::StampedTransform(marker_pose,orb_topic.header.stamp,"ar_marker","/marker_pose"));
}


void PoseRepublisher::CalculateT()
{
	count++;
	if (count >= 10) {
		init_true = true;
	}

	T = ar_pose * orb_pose.inverse();
	
    update_ar = false;
}


void PoseRepublisher::CalculateTruePose()
{
	tf_true_pose = T * orb_pose;

    TruePose.header =  orb_topic.header;
    TruePose.header.frame_id = "ar_marker";
	tf::poseTFToMsg(tf_true_pose,TruePose.pose);
    true_pose_pub.publish(TruePose);

	tf::Vector3 zero_t(0,0,0);
	tf::Quaternion zero_q;
	zero_q.setRPY(0,0,0);
	tf::Transform zero(zero_q, zero_t);
	tb.sendTransform(tf::StampedTransform(zero, orb_topic.header.stamp,"ar_marker","/ar_marker_0"));
	tb.sendTransform(tf::StampedTransform(T, orb_topic.header.stamp,"ar_marker","/orb_map"));
	tb.sendTransform(tf::StampedTransform(orb_pose, orb_topic.header.stamp,"orb_map","/orb_pose"));
    tb.sendTransform(tf::StampedTransform(ar_pose,orb_topic.header.stamp,"ar_marker","/ar_pose"));
    tb.sendTransform(tf::StampedTransform(T, orb_topic.header.stamp,"ar_marker","/Transformation"));
    tb.sendTransform(tf::StampedTransform(tf_true_pose, orb_topic.header.stamp,"ar_marker","/TruePose"));
}