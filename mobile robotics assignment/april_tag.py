import cv2
import apriltag as at
import numpy as np
import gtsam as gt

def calibrate_load(reload, loading_file):
    if reload:
        data = np.load(loading_file)
        
    camera_matrix = data['camera_matrix']
    distortion_coeff = data['distortion_coeff']
    reprojection_error=data['reprojection_error']
    print("Loaded camera matrix:\n", camera_matrix)
    print("Loaded distortion coefficients:\n", distortion_coeff)
    print(f"Reprojection error:{reprojection_error}")
    return camera_matrix, distortion_coeff

def apriltag_detection(image):
    K, d = calibrate_load(reload=True, loading_file="/home/siddarth/ros2ws/src/Robotics_concepts/mobile robotics assignment/camera_calibration/calibration_data.npz")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    calibration_matrix = np.array([[fx], [fy], [cx], [cy]])
    print("Detecting AprilTags...")
    img = cv2.imread(image)
    img=cv2.resize(img,(500,500))
    
    if img is None:
        print("Image not found or unable to read.")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set up AprilTag detector
    options = at.DetectorOptions(families="tag36h11")
    detector = at.Detector(options)
    results = detector.detect(gray)
    print("Total tags detected: {}".format(len(results)))

    corners = None
    for r in results:
        print(f"Tags found:{r.tag_id}")
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
        if tag_id == 0:  # Process only tag 9
            print("Tag DETECTED: {}".format(tag_id))
            # Transformation matrix
            T, _, _ = detector.detection_pose(r, calibration_matrix, 0.159)
            
            
            print(f"Transformation matrix from camera frame to tag frame: \n {T} \n")

            # Extract 2D corners in the image for the detected tag
            corners = r.corners
            

    # Display the result
    cv2.imshow("AprilTags Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
     

    return K, corners

def pose_estimation(K, corners):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    calibration = gt.Cal3_S2(fx, fy, 0, cx, cy)

    # Define the 3D coordinates of the AprilTag's corners in the world frame
    tag_size = 0.159
    half_size = tag_size / 2
    tag_corners_3d = np.array( [
        gt.Point3(-half_size, -half_size, 0),
        gt.Point3(half_size, -half_size, 0),
        gt.Point3(half_size, half_size, 0),
        gt.Point3(-half_size, half_size, 0)
    ])
    
    print(tag_corners_3d)

    # Set up GTSAM factor graph
    graph = gt.NonlinearFactorGraph()
    initial_estimate = gt.Values()
    noise = gt.noiseModel.Isotropic.Sigma(2, 1.0)

    # Add prior factors for each 3D tag corner
    for i, corner_3d in enumerate(tag_corners_3d):
        graph.add(gt.PriorFactorPoint3(i + 1, corner_3d, gt.noiseModel.Constrained.All(3)))
    # print(f"Add prior factors for each 3D tag corner:\n {graph}")
    # Add projection factors for each 2D corner observation
    for i, corner in enumerate(corners):
        corner_2d = gt.Point2(corner[0], corner[1])
        factor = gt.GenericProjectionFactorCal3_S2(
            corner_2d, noise, 0, i + 1, calibration
        )
        graph.add(factor)

    # Initial estimate for the camera pose
    initial_pose = gt.Pose3()
    initial_estimate.insert(0, initial_pose)  

    # Initial estimates for the 3D points of AprilTag corners
    for i, corner_3d in enumerate(tag_corners_3d):
        initial_estimate.insert(i + 1, corner_3d)

    # Optimize the factor graph
    optimizer = gt.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    # Retrieve and print the estimated pose of the camera
    estimated_pose = result.atPose3(0)
    print("Estimated Pose of Camera:\n", estimated_pose)

def main():
    image_path = "/home/siddarth/ros2ws/src/Robotics_concepts/mobile robotics assignment/frame_0.jpg"
    K, corners = apriltag_detection(image_path)
    if corners is not None:
        pose_estimation(K, corners)

if __name__ == "__main__":
    main()
