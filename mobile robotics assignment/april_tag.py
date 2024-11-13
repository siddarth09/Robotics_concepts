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
    img = cv2.resize(img, (500, 500))
    
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
            T, _, _ = detector.detection_pose(r, calibration_matrix, 0.159)
            print(f"Transformation matrix from camera frame to tag frame: \n {T} \n")

            # Extract 2D corners in the image for the detected tag
            corners = r.corners

    # Display the result
    cv2.imshow("AprilTags Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return K, T, corners

def reprojection_error(params, K, P_3d, u_2d):
    """
    Calculate the sum of squared reprojection errors.

    params: flattened array containing the rotation matrix (R) and translation vector (t).
    K: Camera matrix.
    P_3d: 3D points in the world coordinate system.
    u_2d: Observed 2D points in the image plane.
    """
    # Extract R and t from params (flattened array)
    R = params[:9].reshape(3, 3)  # First 9 values represent the rotation matrix (3x3)
    t = params[9:]  # The remaining values represent the translation vector (3,)

    # Initialize error
    total_error = 0
    epsilon = 1e-9  # Small value to avoid divide by zero

    for i in range(len(P_3d)):
        # Project the 3D point to the camera's coordinate system
        p_i = np.dot(R, P_3d[i]) + t

        # Project onto the image plane using the camera matrix
        u_i = np.dot(K, p_i) 
       
        u_i = u_i[:2] / (u_i[2] + epsilon)  

        
        error = np.linalg.norm(u_i - u_2d[i])
        # print(f"error at  point {i}: {error}")


    return error


def pose_estimation(K, T, corners):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    calibration = gt.Cal3_S2(fx, fy, 0, cx, cy)

    # Define the 3D coordinates of the AprilTag's corners in the world frame
    tag_size = 0.159  # Size of the AprilTag in meters
    half_size = tag_size / 2
    tag_corners_3d = np.array([
        [-half_size, -half_size, 0],
        [-half_size, half_size, 0],
        [half_size, half_size, 0],
        [half_size, -half_size, 0]
    ])

    # Extract 2D corners from the image and store them
    u_2d = np.array(corners)

    # Set up optimization
    initial_params = np.hstack([np.eye(3).flatten(), np.zeros(3)])  # Initial guess for [R|t]

    # Optimize the pose (R,t) using the reprojection error function
    result = minimize(reprojection_error, initial_params, args=(K, tag_corners_3d, u_2d), method='BFGS')

    # Extract optimized R and t from the result
    optimized_params = result.x
    optimized_R = optimized_params[:9].reshape(3, 3)
    optimized_t = optimized_params[9:]

    print("Optimized Rotation Matrix:\n", optimized_R)
    print("Optimized Translation Vector:\n", optimized_t)
    print("Final Reprojection Error:", result.fun)


def main():
    image_path = "/home/siddarth/ros2ws/src/Robotics_concepts/mobile robotics assignment/frame_0.jpg"
    K, T, corners = apriltag_detection(image_path)
    if corners is not None:
        pose_estimation(K, T, corners)

if __name__ == "__main__":
    main()
