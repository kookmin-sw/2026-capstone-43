cmake_minimum_required(VERSION 2.8.3)
project(roas_bringup)

find_package(catkin REQUIRED COMPONENTS
)

catkin_package()

install(
  DIRECTORY launch meshes rules urdf config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

