# LiDAR and Camera Sensor fusion for Vehicle detection and Tracking

![Demo](https://github.com/nsteve2407/nd013-c2-fusion-starter/blob/main/img/demo.gif)

This project implements LiDAR and Camera late fusion approach for object detection. Camera images are used for generating 2D detections using an SSD detector trained on the Waymo Open Dataset. Vehicles are detected in the LiDAR point clouds using the Complex YOLO detection framework. An Extented Kalman Filter (EKF) is used to fuse measurements from both these sensors to enable multi-target detection and traking.

#### 2D Object Detector
2D object detections are made using an SSD Detector trained on the Waymo Open Dataset. Currently only using detections for vehicles.

#### Lidar Detector
Complex YOLO is used to detect vehicles in the LIDAR BEV space. The model was pretrained on the KITTI dataset.

#### Fusion and Tracking
Fusion is done using an EKF with a constant velocity motion model. All detections are in the vehicle frame of refernce. Camera intrinsic parameters are used to transform predicted tracks into the pixel coordinate frame. Since this is a nonlinear measurement function we linearize the function at the state mean value by calculating the jacobian matrix. Initial results from Camera and Lidar fused detections on the Waymo Open Dataset are shown below:
![img4](https://github.com/nsteve2407/nd013-c2-fusion-starter/blob/main/img/Step4-RMSE.png)


#### To do :
- Use a bicycle model for motion prediction in the predict step.
- Add additional state variables such as length, width, height and yaw.
- Use better assocaition methods such as GNN/JPDA
- ROS wrappers for real world testing and visualization
