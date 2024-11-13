import numpy as np
import cv2  # OpenCV library for Rodrigues function

# Given camera intrinsic matrix
camera_matrix = np.array([
    [689.53803099, 0, 252.75120826],
    [0, 387.16483042, 244.10383713],
    [0, 0, 1]
])

# Translation and rotation vectors as provided
translation = np.array([
    [1.15877324, -3.88860482, 13.28893989],
    [2.50732289, -4.02784951, 13.58139628],
    [0.62286616, -4.29622831, 11.63751823],
    [2.36722656, -4.35642422, 11.14700267],
    [2.90486793, -3.47755642, 11.9098968],
    [3.95960217, -3.90546111, 12.13971242],
    [-2.13977801, 4.61710163, 15.44743039]
])

rotation_vectors = np.array([
    [-0.80644511, -0.8395277, 1.40059151],
    [-0.31090244, 0.54488106, 1.47436854],
    [0.13721023, 0.56270559, 1.27136907],
    [-0.38163095, -0.40003741, 1.50845834],
    [-0.17085964, -0.14629954, 1.56852997],
    [-0.21913747, -0.20542399, 1.57544579],
    [0.14294946, 0.5411519, -1.53496295]
])

# Project 3D points onto 2D image plane
def project_points(points_3d, K, R, t):
    """
    Projects 3D points onto a 2D image plane given camera calibration parameters.

    Args:
        points_3d (np.ndarray): Nx3 array of 3D points in the world coordinate system.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        R (np.ndarray): 3x3 rotation matrix from the world to the camera coordinate system.
        t (np.ndarray): 3-element translation vector from the world to the camera coordinate system.

    Returns:
        np.ndarray: Nx2 array of 2D projected points in the image plane.
    """
    # Ensure translation is a 3x1 vector for concatenation
    t = t.reshape(3, 1)

    # Construct the 3x4 extrinsic matrix from R and t
    extrinsic_matrix = np.hstack((R, t))

    # Convert points to homogeneous coordinates (Nx4)
    points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Project points to the camera frame
    points_camera_frame = extrinsic_matrix @ points_3d_h.T  # Result is 3xN

    # Project to the image plane using the intrinsic matrix
    points_2d_h = K @ points_camera_frame  # Result is 3xN

    # Convert from homogeneous to 2D coordinates by dividing by the last row
    points_2d = points_2d_h[:2] / points_2d_h[2]

    return points_2d.T  # Nx2 array of 2D points in image coordinates

# Example 3D points (replace these with your actual points)
points_3d = np.array([
    [0, 0, 5],
    [1, 1, 5],
    [-1, -1, 5]
])

# Loop through each translation and rotation vector pair
projected_points = []
for i in range(len(translation)):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rotation_vectors[i])
    t = translation[i]
    
    # Project the points
    points_2d = project_points(points_3d, camera_matrix, R, t)
    projected_points.append(points_2d)
    

print("Projected 2D points for each pose:", projected_points)

