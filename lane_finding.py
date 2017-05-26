# Libraries import
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import line

# To open the image in an external window use %matplotlib qt
# %matplotlib inline

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Camera calibration
def calibrate_camera(path):
    # Read in the calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    # Set up two empty arrays to hold object points and image points from all the images

    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    # Number of corners on the x axis
    nx = 9
    # Number of corners on the y axis
    ny = 6

    # Preapare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (7,5,0)
    objp = np.zeros((ny*nx,3), np.float32) # 6 by 9 points in an array, each with three columns for the x, y and z coordinates of each corner

    # For x and y, use numpy's mgrid function to generate the coordinates I want, z can stay as 0
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates

    # NOTE: mgrid returns the coordinate values for a given grid size and I'll shape those coordinates back
    # into two columns, one for x and one for y

    # To create the image points, I want to look at the distorted calibration image and detect the corners of the board
    # the OpenCV function "findChessboardCorners" returns the corners found in a grayscale image
    # so the next step would be to turn the image into grayscale and then pass it to the findChessboardCorners function
    # The function takes in the grayscale image along with the dimensions of the chessboard corners, in this case 9 by 6
    # last parameter is for any flags, which are none for this case

    for file_name in images:
        # Read in each image
        img = mpimg.imread(file_name)

        # Convert image to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
        # If corners are found, append the points to the image points array
        if ret == True:
            imgpoints.append(corners)
            # And also add the prepared object points (objp) to the object points array
            objpoints.append(objp)

            # Print the corners, just to see the structure, UPDATE:  it's the same as the one on perspective_transform.ipynb
            # Where you decide the source and destination coordinates
            #print('Corners array:',corners)

            # draw the detected corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            plt.imshow(img)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs

# Undistort images
def undistort_image(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# Define the trapezoid form - region of interest
def trapezoid_offset_sizes(image):

    img_size = (image.shape[1], image.shape[0])

    # Capture the x and y size
    height = image.shape[0]
    width = image.shape[1]

    image_x_center = width / 2
    image_y_center = height / 2

    # Next we create our polygon to define which area in the image we are interested in finding lines
    imshape = image.shape
    #
    offset_y = image_y_center * 0.24
    offset_x = image_x_center * 0.10

    bottom_left_offset_x = width * 0.13
    bottom_right_offset_x = width * 0.08

    top_left_offset_x = image_x_center * 0.05
    top_right_offset_x = image_x_center * 0.14
    offset = 0
    bottom_offset_x = width * 0.03

    return img_size, height, width, image_x_center, image_y_center, offset_y, bottom_left_offset_x, bottom_right_offset_x, top_left_offset_x, top_right_offset_x

# Draw the trapezoid lines on the image
def draw_trapezoid_lines(image):

    # Width, height and offsets to help draw the trapezoid
    img_size, height, width, image_x_center, image_y_center, offset_y, bottom_left_offset_x, bottom_right_offset_x, top_left_offset_x, top_right_offset_x = trapezoid_offset_sizes(image)

    # Draw the lines starting from the top to the bottom
    # Line color - red
    color = [255, 0, 0]
    # Line thickness
    thickness = 10
    # Left line
    cv2.line(image, (int(round(0+bottom_left_offset_x)),height), (int(round(image_x_center-top_left_offset_x)), int(round(image_y_center+offset_y))),color,thickness)
    # Right line
    cv2.line(image, (int(round(width-bottom_right_offset_x)),height), (int(round(image_x_center+top_right_offset_x)), int(round(image_y_center+offset_y))),color,thickness)
    # Top line
    cv2.line(image, (int(round(image_x_center-top_left_offset_x)), int(round(image_y_center+offset_y))), (int(round(image_x_center+top_right_offset_x)), int(round(image_y_center+offset_y))), color,thickness )

# Define perspective transform function
def warp(img):

    # Define calibration box in source (original) and destination (desired or warped) coordinates

    # Width, height and offsets to help specify the trapezoid source points
    img_size, height, width, image_x_center, image_y_center, offset_y, bottom_left_offset_x, bottom_right_offset_x, top_left_offset_x, top_right_offset_x = trapezoid_offset_sizes(img)

    w,h = width,height
    x,y = 0.5*w, 0.8*h

    # Four source coordinates
    src = np.float32([[200./1280*w,720./720*h],
                  [453./1280*w,547./720*h],
                  [835./1280*w,547./720*h],
                  [1100./1280*w,720./720*h]])
    # Four desired coordinates -- where the warped image will be placed
    dst = np.float32([[(w-x)/2.,h],
                  [(w-x)/2.,0.82*h],
                  [(w+x)/2.,0.82*h],
                  [(w+x)/2.,h]])

    # Compute the perspective transform, M
    # This function takes in four source points and four destination points and returns the mapping perspective matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Could compute the inverse also by swapping the input parameters - use to unwarp the image
    M_inverted = cv2.getPerspectiveTransform(dst, src)

    # Apply the transform M to the original image to get the warped image by calling the warp perspective function
    # This function takes in the image, the perspective matrix M, the size we want the warped image to be, and
    # how to interpolate points - fill in missing points as it warps an image - use - linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M_inverted

# Color and gradient Threshold
def hls_sobel_x_gradient(warped_image):
    image = warped_image
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary, color_binary

def hls_lab_threshold(warped_image, hls_thresh=(220, 255), lab_thresh=(190,255)):
    # HLS

    # Convert to HLS color space
    hls = cv2.cvtColor(warped_image, cv2.COLOR_RGB2HLS)
    hls_l = hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    # Apply a threshold to the L channel
    hls_binary_output = np.zeros_like(hls_l)
    hls_binary_output[(hls_l > hls_thresh[0]) & (hls_l <= hls_thresh[1])] = 1

    # LAB

    # Convert to LAB color space
    lab = cv2.cvtColor(warped_image, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # Apply a threshold to the B channel
    lab_binary_output = np.zeros_like(lab_b)
    lab_binary_output[((lab_b > lab_thresh[0]) & (lab_b <= lab_thresh[1]))] = 1

    # Combined

    # Combine HLS and Lab B channel thresholds
    combined_hls_lab_b = np.zeros_like(lab_binary_output)
    combined_hls_lab_b[(hls_binary_output == 1) | (lab_binary_output == 1)] = 1

    combined_binary = combined_hls_lab_b

    return combined_binary

#
def sliding_window_search(combined_binary):

    # To decide explicitly which pixels are part of the lines
    # and which belong to the left line and which belong to the right line
    # take a histogram of all the columns on the lower half of the binary image
    histogram = np.sum(combined_binary[combined_binary.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((combined_binary, combined_binary, combined_binary))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(combined_binary.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = combined_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = combined_binary.shape[0] - (window+1)*window_height
        win_y_high = combined_binary.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit, right_fit = [[0,0,0], [0,0,0]]
    if len(leftx) != 0 and len(rightx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        #print(left_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, combined_binary.shape[0]-1, combined_binary.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    #return out_img, ploty, left_fitx, right_fitx, left_fit, right_fit, left_lane_inds, right_lane_inds
    return out_img, ploty, left_fitx, right_fitx, left_fit, right_fit

# Skip sliding window search
def skip_sliding_window_search(combined_binary):
    # Assume you now have a new warped binary image
    # from the next frame of video
    # It's now much easier to find line pixels!
    binary_warped = combined_binary

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit, right_fit = [[0,0,0], [0,0,0]]
    if len(leftx) != 0 and len(rightx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result, ploty, left_fitx, right_fitx

# Calculate the radius of the curvature
def calculate_curvature_radius2(combined_binary, ploty, left_fit, right_fit):

    height = combined_binary.shape[0]
    # meters per pixel in y dimension
    ym_per_pix = 30/height
    # meters per pixel in x dimension
    xm_per_pix = 3.7/700

    y_points = np.linspace(0, height-1, height)
    y_real_world = y_points*ym_per_pix
    y_eval = np.max(y_real_world)

    # Left line
    x_points_left = left_fit[0]*y_points**2 + left_fit[1]*y_points + left_fit[2]
    x_real_world_left = x_points_left*xm_per_pix

    # Right Line
    x_points_right = right_fit[0]*y_points**2 + right_fit[1]*y_points + right_fit[2]
    x_real_world_right = x_points_right*xm_per_pix

    # Fit polynomial using the real world values
    fit_cr_left = np.polyfit(y_real_world, x_real_world_left, 2)
    fit_cr_right = np.polyfit(y_real_world, x_real_world_right, 2)

    # Curvature computation
    curverad_left = ((1 + (2*fit_cr_left[0]*y_eval+ fit_cr_left[1])**2)**1.5) / np.absolute(2*fit_cr_left[0])
    curverad_right = ((1 + (2*fit_cr_right[0]*y_eval+ fit_cr_right[1])**2)**1.5) / np.absolute(2*fit_cr_right[0])

    # Calculate distance from center
    vehicle_position = combined_binary.shape[1]/2

    left_fit_x_int = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
    right_fit_x_int = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]

    lane_center_position = (right_fit_x_int + left_fit_x_int) /2

    distance_from_center = (vehicle_position - lane_center_position) * xm_per_pix

    return curverad_left, curverad_right, distance_from_center

def draw_detected_lane_lines(combined_binary, image, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    warped = combined_binary
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Draw the lane lines
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,0), thickness=25) # Red left line
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,0,255), thickness=25)# Blue right line

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result

# Draw the curvature radius and distance of the car from the center of the lane on to the image with the lane
# lines detected
def draw_curvature_radius_information(lane_detected_image, left_curverad, right_curverad, distance_from_center):

    new_img = np.copy(lane_detected_image)
    # Define font style
    font = cv2.FONT_HERSHEY_TRIPLEX
    # Sum the left and right curve radius
    curve_radius = left_curverad + right_curverad
    text = 'Curve radius: ' + '{:04.2f}'.format(curve_radius) + 'm'
    cv2.putText(new_img, text, (40,70), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
    position = ''
    if distance_from_center > 0:
        position = 'right'
    elif distance_from_center < 0:
        position = 'left'
    abs_distance_from_center = abs(distance_from_center)
    text = 'Vehicle is {:04.3f}'.format(abs_distance_from_center) + 'm ' + position + ' of center'
    cv2.putText(new_img, text, (40,120), font, 1.5, (255,255,255), 2, cv2.LINE_AA)

    return new_img

def process_image(image):
    # Undistort the image
    image = undistort_image(image, mtx, dist)

    # Perform perspective transformation
    warped_im, Minv = warp(image)

    # Perform color and gradients thresholding

    # HLS and Sobelx Gradient combined
    #combined_binary, color_binary = hls_sobel_x_gradient(warped_im)

    # HLS and LAB thresholding combined
    combined_binary = hls_lab_threshold(warped_im)

    # Perform the sliding window search when unsure
    out_img, ploty, left_fitx, right_fitx, left_fit, right_fit = sliding_window_search(combined_binary)
    #out_img, ploty, left_fitx, right_fitx, left_fit, right_fit, left_lane_inds, right_lane_inds = sliding_window_search(combined_binary)

    # NEED TO ARRANGE THIS IN A BETTER FASHION
    # If you are pretty sure where that the sliding window search is working good, you can skip it for the next frame
    # out_img, ploty, left_fitx, right_fitx = skip_sliding_window_search(combined_binary)

    # It's now ready to draw the detected lane lines on the original undistorted image
    img = draw_detected_lane_lines(combined_binary, image, left_fitx, right_fitx, ploty, Minv)

    # Calculate the radius of the curvature
    #left_curverad, right_curverad, distance_from_center = calculate_curvature_radius(combined_binary, ploty, left_fit, right_fit, leftx, rightx, lefty, righty)
    #left_curverad, right_curverad, distance_from_center = calc_curv_rad_and_center_dist(combined_binary, left_fit, right_fit, left_lane_inds, right_lane_inds )
    left_curverad, right_curverad, distance_from_center = calculate_curvature_radius2(combined_binary, ploty, left_fit, right_fit)
    # And finally draw the radius of the curvature and the distance of the car from the center
    img = draw_curvature_radius_information(img, left_curverad, right_curverad, distance_from_center)

    return img

def detect_lanes_on_images(distorted_images_path, output_path):
    images = glob.glob(distorted_images_path)
    path = output_path+"image"
    extension = ".jpg"
    i = 0
    for file_name in images:
        image = mpimg.imread(file_name)
        # Perform perspective transform, color transform, gradient thresholding
        # Sliding window search and draw the detected lane
        left_line = line.Line()
        right_line = line.Line()
        processed_image = process_image(image)
        # Save the resulting image to file
        mpimg.imsave(path+str(i)+extension,processed_image)
        i = i+1

def detect_lanes_on_video(video_path, output_path):
    output = output_path
    clip = VideoFileClip(video_path)
    # Perform perspective transform, color transform, gradient thresholding
    # Sliding window search and draw the detected lane
    left_line = line.Line()
    right_line = line.Line()
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output, audio=False)

def save_undistorted_chessboard_image():
    # Save an undistorted chessboard image to file
    # Read in the image
    img = mpimg.imread('camera_cal/calibration1.jpg')
    # Undistort the image
    chessboard_undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    # Save the undistorted image to file
    mpimg.imsave('output_images/undistorted_chessboard.jpg', chessboard_undistorted)
