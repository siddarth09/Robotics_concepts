import cv2
import apriltag as at
import numpy as np
import gtsam as gt
from scipy.optimize import minimize

def calibrate_load(reload, loading_file):
    if reload:
        data = np.load(loading_file)
        
    camera_matrix = data['camera_matrix']
    distortion_coeff = data['distortion_coeff']
    reprojection_error = data['reprojection_error']
    rotation = data["rotation"]
    translation = data["translation"]
    print("Loaded camera matrix:\n", camera_matrix)
    print("Loaded distortion coefficients:\n", distortion_coeff)
    print(f"Reprojection error:{reprojection_error}")
    return camera_matrix, distortion_coeff, reprojection_error, rotation, translation

def apriltag_detection(image):
    K, d, err_r, rot, trans = calibrate_load(reload=True, loading_file="/home/siddarth/ros2ws/src/Robotics_concepts/mobile robotics assignment/camera_calibration/calibration_data.npz")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    calibration_matrix = np.array([[fx], [fy], [cx], [cy]])
    print("Detecting AprilTags...")
    img = cv2.imread(image)
    # img = cv2.resize(img, (500, 500))
    
    if img is None:
        print("Image not found or unable to read.")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set up AprilTag detector
    options = at.DetectorOptions(families="tag36h11")
    detector = at.Detector(options)
    results = detector.detect(gray)
    
    corners = None
    for r in results:
        (ptA, ptB, ptC, ptD) = r.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))

        cv2.line(img, ptA, ptB, (0, 255, 0), 2)
        cv2.line(img, ptB, ptC, (0, 255, 0), 2)
        cv2.line(img, ptC, ptD, (0, 255, 0), 2)
        cv2.line(img, ptD, ptA, (0, 255, 0), 2)

        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
        
        tag_id = r.tag_id
        if tag_id == 0:  # Process only tag 0
            print("Tag DETECTED: {}".format(tag_id))
            # Transformation matrix
            T, _, _ = detector.detection_pose(r, calibration_matrix, 0.01)
            print(f"Transformation matrix from camera frame to tag frame: \n {T} \n")

            # Extract 2D corners in the image for the detected tag
            corners = r.corners
    
    # Display the result
    cv2.imshow("AprilTags Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return K, T,err_r,corners

def pose_estimation(K, T, err,corners):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    calibration = gt.Cal3_S2(fx, fy, 0, cx, cy)

    # Define the 3D coordinates of the AprilTag's corners in the world frame
    tag_size = 0.01 # Size of the AprilTag in meters
    half_size = tag_size / 2
    tag_corners_3d = np.array([
        [-half_size, -half_size, 0],
        [-half_size, half_size, 0],
        [half_size, half_size, 0],
        [half_size, -half_size, 0]
    ])
    
    # Set up GTSAM factor graph
    graph = gt.NonlinearFactorGraph()
    noise = gt.noiseModel.Isotropic.Sigma(3,1.0)  # Noise model for pose
    initial_pose = gt.Pose3(gt.Rot3.RzRyRx(0.3,0.2,0.2), gt.Point3(0.0075,0.0075,-0.25))  # Initial guess for the pose
    pose_symbol=gt.symbol('x',0)
    # graph.add(gt.PriorFactorPose3(0, initial_pose, noise))

    # Create initial estimate for the camera pose
    initial_estimate = gt.Values()
    initial_estimate.insert(pose_symbol, initial_pose)
    # graph.add(gt.PriorFactorPose3(pose_symbol, initial_pose, noise))

    # Creating Initial estimates for 3D points
    for i, corner3d in enumerate(tag_corners_3d):
        point_symbol=gt.symbol('p',i)
        initial_estimate.insert(point_symbol,corner3d)
        graph.add(gt.PriorFactorPoint3(point_symbol,corner3d,noise))
    corner2d_arr = [gt.Point2(corner[0], corner[1]) for corner in corners]
    # Add projection factors for each pair of 2D-3D correspondences
    for i, corner2d in enumerate(corners):
        corner2d_array = gt.Point2(corner2d[0],corner2d[1])
        point_symbol=gt.symbol('p',i)
        factor = gt.GenericProjectionFactorCal3_S2(
            corner2d_array, 
            gt.noiseModel.Isotropic.Sigma(2, 10.0), 
            pose_symbol, 
            point_symbol, 
            calibration
        )
        graph.add(factor)
        
        
    params=gt.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setVerbosity("TERMINATION")
    params.setlambdaInitial(1e-9)
    # Optimize using Levenberg-Marquardt optimizer
    optimizer = gt.LevenbergMarquardtOptimizer(graph, initial_estimate,params)
    result = optimizer.optimize()

    # Get the optimized pose
    optimized_pose = result.atPose3(pose_symbol)
    print("Optimized Pose:\n", optimized_pose)
    print(f"The reprojection error is {graph.error(result)}")

   
def main():
    image_path = "/home/siddarth/ros2ws/src/Robotics_concepts/mobile robotics assignment/frame_0.jpg"
    K, T,err,corners = apriltag_detection(image_path)
    if corners is not None:
        pose_estimation(K, T, err,corners)

if __name__ == "__main__":
    main()
