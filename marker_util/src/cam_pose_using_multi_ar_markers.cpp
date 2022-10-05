//TODO: use marker_util

#include <ros/ros.h>
#include <ar_track_alvar_msgs/AlvarMarkers.h>
#include <tf/tf.h>

geometry_msgs::PoseStamped cam_pose;
tf::Transform T01, T02;

tf::Transform tf_average(std::vector<tf::Transform> &t)
{
    double px=0, py=0, pz=0, qx=0, qy=0, qz=0, qw=0;
    double lx=0, ly=0, lz=0, theta=0;
    for (size_t i = 0; i < t.size(); i++)
    {
        // ROS_INFO("q: (%lf, %lf, %lf, %lf)", t[i].getRotation().getX(), t[i].getRotation().getY(), t[i].getRotation().getZ(), t[i].getRotation().getW());
        px += t[i].getOrigin().getX(); py += t[i].getOrigin().getY(); pz += t[i].getOrigin().getZ();
        double a = std::acos(t[i].getRotation().getW());
        theta += 2*a;
        if (a==0)
        {
            lx += 0; ly += 0; lz += 0;
        }
        else
        {
            lx += t[i].getRotation().getX()/std::sin(a); ly += t[i].getRotation().getY()/std::sin(a); lz += t[i].getRotation().getZ()/std::sin(a);
        }
        // ROS_INFO("%zu[ p: (%lf, %lf, %lf) ",i, t[i].getOrigin().getX(), t[i].getOrigin().getY(), t[i].getOrigin().getZ());
        // ROS_INFO("q: (%lf, %lf, %lf, %lf)]", t[i].getRotation().getX(), t[i].getRotation().getY(), t[i].getRotation().getZ(), t[i].getRotation().getW());
    }
    // ROS_INFO("q: (%lf, %lf, %lf, %lf)", lx, ly, lz, theta);
    px = px / t.size();py = py / t.size();pz = pz / t.size();
    theta = theta / t.size();
    lx = lx / t.size();ly = ly / t.size();lz = lz / t.size();
    qx = lx*std::sin(theta/2);qy = ly*std::sin(theta/2);qz = lz*std::sin(theta/2);qw = std::cos(theta/2);
    tf::Transform t_avg;
    t_avg.setOrigin(tf::Vector3(px, py, pz));
    // ROS_INFO("q: (%lf, %lf, %lf, %lf)", qx, qy, qz, qw);
    t_avg.setRotation(tf::Quaternion(qx, qy, qz, qw));
    return t_avg;
}

void arMarkerCB(const ar_track_alvar_msgs::AlvarMarkers::ConstPtr &msg)
{
    ar_track_alvar_msgs::AlvarMarkers ar = *msg;
    geometry_msgs::PoseStamped marker0, marker1, marker2;
    bool rec0=false, rec1=false, rec2=false;
    tf::Transform ar_pose;
    for (size_t i = 0; i < ar.markers.size(); i++)
    {
        if (ar.markers[i].id == 0)
        {
            ROS_INFO("marker0[%zu]", i);
            marker0 = ar.markers[i].pose;
            rec0=true;
        }
        if (ar.markers[i].id == 1)
        {
            ROS_INFO("marker1[%zu]", i);
            marker1 = ar.markers[i].pose;
            rec1=true;
        }
        if (ar.markers[i].id == 2)
        {
            ROS_INFO("marker2[%zu]", i);
            marker2 = ar.markers[i].pose;
            rec2=true;
        }
    }

    std::vector<tf::Transform> poses;
    tf::Transform marker0_pose, marker1_pose, marker2_pose;

    if (rec0)
    {
        tf::poseMsgToTF(marker0.pose, marker0_pose);
        marker0_pose = marker0_pose.inverse();
        poses.push_back(marker0_pose);
    }
    if (rec1)
    {
        tf::poseMsgToTF(marker1.pose, marker1_pose);
        marker1_pose = T01*marker1_pose.inverse();
        poses.push_back(marker1_pose);
    }
    if (rec2)
    {
        tf::poseMsgToTF(marker2.pose, marker2_pose);
        marker2_pose = T02*marker2_pose.inverse();
        poses.push_back(marker2_pose);
    }

    tf::Transform t = tf_average(poses);
    ROS_WARN("t[ p: (%lf, %lf, %lf) ", t.getOrigin().getX(), t.getOrigin().getY(), t.getOrigin().getZ());
    ROS_WARN("q: (%lf, %lf, %lf, %lf)]", t.getRotation().getX(), t.getRotation().getY(), t.getRotation().getZ(), t.getRotation().getW());

    // for (size_t i = 0; i < poses.size(); i++)
    // {
    //     ROS_INFO("%zu[ p: (%lf, %lf, %lf) ",i, poses[i].getOrigin().getX(), poses[i].getOrigin().getY(), poses[i].getOrigin().getZ());
    //     ROS_INFO("q: (%lf, %lf, %lf, %lf)]", poses[i].getRotation().getX(), poses[i].getRotation().getY(), poses[i].getRotation().getZ(), poses[i].getRotation().getW());
    // }
    
    tf::poseTFToMsg(t,cam_pose.pose);
    cam_pose.header = ar.header;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "cam_pose_using_ar_marker");
    ROS_INFO("cam_pose_using_ar_marker start");
    
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    std::string ar_topic, pose_topic;
    ar_topic = "/ar_pose_marker";
    pose_topic = "/marker_pose";

    T01.setOrigin(tf::Vector3(-2.,2.,0.));
    T01.setRotation(tf::createQuaternionFromRPY(0., 0., 0.));
    T02.setOrigin(tf::Vector3(-0.5,0.,0.));
    T02.setRotation(tf::createQuaternionFromRPY(0., 0., 0.));

    ros::Subscriber ar_sub = nh.subscribe(ar_topic, 1, &arMarkerCB);
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>(pose_topic, 1);

    ros::Rate rate(15);
    while (ros::ok())
    {
        ros::spinOnce();
        pose_pub.publish(cam_pose);
        rate.sleep();
    }

    ROS_INFO("cam_pose_using_ar_marker finished");

    return 0;
}