import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

def calibrate(imgPath,showPics=True):
    
    if len(imgPath) == 0:
        print("No images found in the specified directory.")
        return

    # Initialize calibration parameters
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    objp = np.zeros((8 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane
    
    for file in imgPath:
        img = cv.imread(file)
        if img is None:
            print(f"Failed to load image: {file}")
            continue
        # img = cv.resize(img, (512,512))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Finding chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (8, 6), None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            if showPics:
                cv.drawChessboardCorners(img, (8, 6), corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(500)
                
        # cv.imshow('img', img)
        # cv.waitKey(500)
    
    cv.destroyAllWindows()

    # Verify if there are any valid points for calibration
    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("No valid points found for calibration.")
        return

    # Perform calibration
    ret, camera_matrix, distortion_coeff, rotation, translation = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", distortion_coeff)
    print("Reprojection Error:", ret)
    
    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rotation[i], translation[i],camera_matrix, distortion_coeff)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )

    h, w = img.shape[:2]
        # Get optimal new camera matrix
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w, h), 1, (w, h))
    
    print(f"Comparing martix : \n {camera_matrix}\n and \n {new_camera_matrix}")
    np.savez("calibration_data.npz", 
         camera_matrix=camera_matrix, 
         distortion_coeff=distortion_coeff, 
         rotation=rotation, 
         translation=translation, 
         reprojection_error=ret)

    print("Calibration data saved.")
    # Return the camera matrix and distortion coefficients for further use
    return camera_matrix, distortion_coeff

def removeDistortion(imgPath, camera_matrix, distortion_coeff):
    """
    This function takes an image and removes distortion using the camera matrix and distortion coefficients.
    """
    for file in imgPath:
        img=cv.imread(file)
        img=cv.resize(img, (500,500))
        h, w = img.shape[:2]
        # Get optimal new camera matrix
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w, h), 1, (w, h))

        # Undistort the image
        undistorted_img = cv.undistort(img, camera_matrix, distortion_coeff, None, new_camera_matrix)
        
        # Crop the image if necessary
        x, y, w, h = roi
        undistorted_img = undistorted_img[y:y+h, x:x+w]
        
    return img,undistorted_img



def runCalibrate():
    # Run calibration to get the camera matrix and distortion coefficients
    
    root = os.getcwd()
    calibdir = os.path.join(root, 'calibration_images/calibration_images')
    imgPath = glob.glob(os.path.join(calibdir, '*.JPEG'))
    
    result = calibrate(imgPath,showPics=True)
    if result is None:
        print("Calibration failed. Cannot proceed to undistortion.")
        return
    
    camera_matrix, distortion_coeff = result

    # # Load a test image for undistortion
    # test_image_path = "path/to/your/test_image.JPEG"  # Replace with actual test image path
    # test_img = cv.imread(test_image_path)
    # if test_img is None:
    #     print("Failed to load test image for undistortion.")
    #     return

    # Remove distortion from the test image
    distorted,undistorted_img = removeDistortion(imgPath, camera_matrix, distortion_coeff)

    # Show the original and undistorted images
    cv.imshow("Original Image", distorted)
    cv.imshow("Undistorted Image", undistorted_img)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    runCalibrate()
