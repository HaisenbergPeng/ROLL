
loam_version: a iros version
master: a much simplified mapping module

Environments:
Ubuntu 16.04, ROS kinetic
PCL 1.7
Eigen 3.3.7
GTSAM 4.0.2 (-DGTSAM_BUILD_WITH_MARCH_NATIVE = OFF)

rviz localization need PCL 1.8 for TEASER (preinstalled in '/usr/local'):
1. TEASER installation: cmake -DBUILD_TESTS=OFF -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_DOC=OFF -DBUILD_TEASER_FPFH=ON -DCMAKE_INSTALL_PREFIX= /usr/local ..
2. PCL 1.8 needs to be installed in a different directory from PCL 1.7




