#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <ar_track_alvar_msgs/AlvarMarkers.h>

#include <tf/tf.h>
#define PI    3.141592653589793238

int main(int argc, char **argv)
{
	ros::init(argc, argv, "test_pose_republisher");
    ros::NodeHandle nh;
	
    ros::Publisher orb_pub = nh.advertise<geometry_msgs::PoseStamped>("/orb_slam2_mono/pose", 10);
    ros::Publisher ar_pub = nh.advertise<ar_track_alvar_msgs::AlvarMarkers>("/ar_pose_marker", 10);

    geometry_msgs::PoseStamped orb_topic;
    //set  orb_topic
    orb_topic.header.frame_id ="orb_map";
    orb_topic.pose.position.x = 1.;
    orb_topic.pose.orientation.w = 1.;

    //
    ar_track_alvar_msgs::AlvarMarker ar_topic_0;
    geometry_msgs::PoseStamped ar_pose_0;
    ar_track_alvar_msgs::AlvarMarkers ar_topic;
    //set ar_pose_0
    ar_pose_0.header.frame_id ="camera";
    ar_pose_0.pose.position.x = -2.;
    ar_pose_0.pose.position.y = -1.;
    ar_pose_0.pose.orientation.w = 1.;
    //
    tf::Vector3 t1(ar_pose_0.pose.position.x, ar_pose_0.pose.position.y, ar_pose_0.pose.position.z);
    tf::Quaternion q1(ar_pose_0.pose.orientation.x, ar_pose_0.pose.orientation.y, ar_pose_0.pose.orientation.z, ar_pose_0.pose.orientation.w);
    tf::Transform p1_t(q1,t1);
    
    /*tf::Vector3 m(0,0,0);
    tf::Quaternion w;
    w.setRPY(0,PI/2,0); 
	tf::Transform a(w,m);

    p1_t = a * p1_t;

    tf::Vector3 p(0,0,0);
    tf::Quaternion q;
    q.setRPY(-PI/2,0,0); 
	tf::Transform t(q,p);

    p1_t = t * p1_t;
*/

    geometry_msgs::Pose ar_pose_;
	tf::poseTFToMsg(p1_t,ar_pose_);
	std::cout << "ar_pose: " << ar_pose_ <<std::endl;

    tf::poseTFToMsg(p1_t,ar_pose_0.pose);



    ar_topic_0.pose = ar_pose_0;
    ar_topic.markers.push_back(ar_topic_0);


	ros::Rate rate(10);
	while(ros::ok())
	{
		ros::spinOnce();
		orb_pub.publish(orb_topic);
        ar_pub.publish(ar_topic);	
		rate.sleep();
	}
	return 0;
}