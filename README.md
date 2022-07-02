## 1. ROLL
![frameworkV10](https://user-images.githubusercontent.com/72208396/176880714-de2cf865-fdee-4796-aadd-d77a5af7d719.png)
ROLL is a LiDAR-based algorithm that can provide robust and accurate localization performance against long-term scene changes. 

* We propose a robust LOAM-based global matching module incorporating temporary mapping, which can prevent localization failures in areas with significant scene changes or insufficient map coverings. The temporary map can be merged onto the global map once matching is reliable again.
* We extend a fusion scheme to trajectories from LIO and noisy global matching. By implementing a consistency check on the derived odometry drift, we successfully prevent the optimization results from going out of bounds.

Related paper: [ROLL: Long-Term Robust LiDAR-based Localization With Temporary Mapping in Changing Environments](https://arxiv.org/abs/2203.03923). 
The paper is now accepted by IROS 2022

## 2. Prerequisites
* Ubuntu 16.04, ROS kinetic
* PCL 1.7
* Eigen 3.3.7
* GTSAM 4.0.2 (-DGTSAM_BUILD_WITH_MARCH_NATIVE = OFF)
* Ceres 2.0 (It is optional since now the pose graph optimization is performed with GTSAM)


## 3. Build
```
mkdir catkin_ws/src -p && cd catkin_ws/src
git clone https://github.com/HaisenbergPeng/ROLL.git
git clone https://github.com/HaisenbergPeng/FAST_LIO.git
cd ../.. && catkin_make
```

## 4. Run
1. [Download NCLT datasets](http://robots.engin.umich.edu/nclt/)

    We need all data except for Image and Hokuyo. Extract all files in one directory.
2. Generate rosbag with the original data

    Change the variables "record_time" and "root_dir", and arrange files in directory "root_dir" as following:
```
├── cov_2012-01-15.csv
├── gps.csv
├── gps_rtk.csv
├── gps_rtk_err.csv
├── groundtruth_2012-01-15.csv
├── kvh.csv
├── ms25.csv
├── ms25_euler.csv
├── odometry_cov_100hz.csv
├── odometry_cov.csv
├── odometry_mu_100hz.csv
├── odometry_mu.csv
├── README.txt
├── velodyne_hits.bin
└── wheels.csv
```
```
cd src/scripts
python nclt_data2bag_BIN.py
```
Then a rosbag named "2012-01-15_bin.bag" will be generated.

3. Building a map with NCLT ground truth
```
roslaunch roll GTmapping_nclt.launch
rosbag play <root_dir>/2012-01-15_bin.bag --clock
```

4. Localization test

    By default, the algorithm will get the initial pose from topic "ground_truth". If it cannot get such a topic,
 it load initial pose from variable "initialGuess".
```
roslaunch roll loc_nclt.launch
rosbag play <root_dir>/<another_bag> --clock
```

5. Evaluation

    All evaluations were performed with matlab scripts, which are open-sourced as well 

## 5. Acknoledgements
Thanks for LOAM, [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM), [FAST-LIO2](https://github.com/hku-mars/FAST_LIO)
