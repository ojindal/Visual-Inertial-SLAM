Implemented visual-inertial simultaneous localization and mapping (SLAM) using an extended Kalman filter (EKF) in Python. Provided data: Synchronized measurements from an inertial measurement unit (IMU) and a stereo camera and the intrinsic camera calibration and the extrinsic calibration between the two sensors, specifying the transformation from the IMU to the left camera frame.

There are total four code files(.py format):

1. pr3_utils.py
2. part_a.py
3. part_b.py
4. part_c.py

- First one contains some utility functions that are used in rest of the code files.

- Second one contains solution of IMU Localization via EKF Prediction.

- Third one contains solution of Landmark Mapping via EKF Update.

- Fourth one contains solution of Visual SLAM.
