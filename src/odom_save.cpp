#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include<geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include<fstream>
// // rosrun kloam odom_save_node /home/binpeng/Documents/LIO-SAM/test/ odomW.txt odomL.txt
class OdomTransform{
  private:
    ros::NodeHandle n;
    ros::Subscriber subWheel;
    ros::Subscriber subLiosam;
    ros::Publisher pub;
  public:
    std::vector<nav_msgs::Odometry> odomVecW;
    std::vector<nav_msgs::Odometry> odomVecL;
    OdomTransform(){
      // this指针非常重要
      subWheel = n.subscribe("odom", 5, &OdomTransform::odomCallback,this);	
      subLiosam = n.subscribe("/kloam/mapping/odometry", 5, &OdomTransform::liosamCallback,this);
    }
    ~OdomTransform(){};
    void odomCallback (const nav_msgs::Odometry& odom_msgW)
    {
      odomVecW.push_back(odom_msgW);
    }
    void liosamCallback (const nav_msgs::Odometry& odom_msgL)
    {
      odomVecL.push_back(odom_msgL);
    }
};

int main(int argc, char** argv){
  ros::init(argc, argv, "odometry_save");
  std::cout<<"odom_save node initialized"<<std::endl;
  OdomTransform OT;
  ros::spin();  

  if (OT.odomVecL.empty() or OT.odomVecW.empty()) {
    std::cout<<"some odom vecs are empty"<<std::endl; 
  }
  else
  {
    int msgNumL = OT.odomVecL.size();
    int msgNumW = OT.odomVecW.size();
    std::cout<<"****************Saving "<<msgNumW <<" wheel encoder odom messages******************"<<std::endl;
    std::string foldername=argv[1];  // common folder
    // For wheel encoder odometry
    std::string filenameW = foldername+argv[2];
    std::ofstream fileW(filenameW);
    if (!fileW.is_open()) {
        std::cout<<filenameW + " couldn't be opened"<<std::endl;
        return 0;
    }
    for (int i = 0; i < msgNumW; i++)
    {
      nav_msgs::Odometry odom_msg = OT.odomVecW[i];
      // double roll,pitch,yaw;
      double x = odom_msg.pose.pose.position.x;
      double y = odom_msg.pose.pose.position.y;
      fileW<<x<<" "<<y<<"\n";
      // tf::Quaternion quat;
      // tf::quaternionMsgToTF(odom_msg.pose.pose.orientation, quat);  
      // tf::Matrix3x3(quat).getRPY(roll, pitch, yaw); 
    }
    fileW.close();

    // For kloam odometry
    std::cout<<"****************Saving "<<msgNumL <<" kloam odom messages******************"<<std::endl;
    std::string filenameL = foldername+argv[3];
    std::ofstream fileL(filenameL);
    if (!fileL.is_open()) {
        std::cout<<filenameL + " couldn't be opened"<<std::endl;
        return 0;
    }
    for (int i = 0; i < msgNumL; i++)
    {
      nav_msgs::Odometry odom_msg = OT.odomVecL[i];
      double x = odom_msg.pose.pose.position.x;
      double y = odom_msg.pose.pose.position.y;
      fileL<<x<<" "<<y<<"\n";
    }
    fileL.close();

    std::cout<<"****************Done Saving odom messages******************"<<std::endl;
    
    return 0;
  }
}