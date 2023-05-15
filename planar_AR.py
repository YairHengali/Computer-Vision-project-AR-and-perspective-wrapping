# ======= imports
import cv2
import numpy as np
import mesh_renderer

# === template image keypoint and descriptors
t_im = cv2.imread("template_img.jpg")
t_im_gray = cv2.cvtColor(t_im, cv2.COLOR_BGR2GRAY)

dim = (t_im.shape[1], t_im.shape[0])

feature_extractor = cv2.SIFT_create()
t_kp, t_desc = feature_extractor.detectAndCompute(t_im_gray, None)

# ===== video input, output and metadata

#calibration:
calib_video = "calibration_video.mp4"
cali_cap = cv2.VideoCapture(calib_video)
cali_ret, cali_frame = cali_cap.read()

height, width, layers = cali_frame.shape
square_size = 1.6
pattern_size = (9, 6)
frame_num = 0

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
obj_points = []
img_points = []

while(cali_ret):
    frame_num += 1
    if frame_num % 40 == 0:
        cali_frame_gray = cv2.cvtColor(cali_frame, cv2.COLOR_RGB2GRAY) 
        found, corners = cv2.findChessboardCorners(cali_frame_gray, pattern_size)
        if not found:
            print("chessboard not found")
            continue

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

    cali_ret, cali_frame = cali_cap.read()

cali_cap.release()


rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (width, height), None, None)

#input video:
video_name = "input_video.mp4"
cap = cv2.VideoCapture(video_name)
ret, frame = cap.read()
height, width, layers = frame.shape
frameSize=(width, height)
fps = cap.get(cv2.CAP_PROP_FPS)
video_out = cv2.VideoWriter('output_video_part2.avi',fourcc=cv2.VideoWriter_fourcc(*'DIVX'),fps=fps, frameSize=frameSize)

renderer = mesh_renderer.MeshRenderer(camera_matrix, width, height,"Rabbit\Rabbit.obj")


# ========== run on all frames
while(ret):
    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    f_kp, f_desc = feature_extractor.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(f_desc, t_desc, k=2)

    good_f_kp = np.array([f_kp[m.queryIdx].pt for m in np.asarray(matches)[:,0]])
    good_t_kp = np.array([t_kp[m.trainIdx].pt for m in np.asarray(matches)[:,0]])

    # ======== find homography
    # also in SIFT notebook
    H, masked = cv2.findHomography(good_t_kp, good_f_kp, cv2.RANSAC, 5.0)

    # ++++++++ take subset of keypoints that obey homography (both frame and reference)
    # this is at most 3 lines- 2 of which are really the same
    # HINT: the function from above should give you this almost completely
    used_f_kp = good_f_kp[masked.ravel() == 1]
    used_t_kp = good_t_kp[masked.ravel() == 1] 


    # ++++++++ solve PnP to get cam pose (r_vec and t_vec)
    # `cv2.solvePnP` is a function that receives:
    # - xyz of the template in centimeter in camera world (x,3)
    # - uv coordinates (x,2) of frame that corresponds to the xyz triplets
    # - camera K
    # - camera dist_coeffs
    # and outputs the camera pose (r_vec and t_vec) such that the uv is aligned with the xyz.
    #
    # NOTICE: the first input to `cv2.solvePnP` is (x,3) vector of xyz in centimeter- but we have the template keypoints in uv
    # because they are all on the same plane we can assume z=0 and simply rescale each keypoint to the ACTUAL WORLD SIZE IN CM.
    # For this we just need the template width and height in cm.
    #
    # this part is 2 rows

    obj_width_cm = 22
    obj_height_cm = 22
    used_t_kp_cm =  used_t_kp / dim * (22, 22)
    used_t_kp_cm_3d = np.column_stack((used_t_kp_cm, np.zeros(used_t_kp_cm.shape[0])))
    object_points = np.array(used_t_kp_cm_3d)
    imagePoints = np.array(used_f_kp)
    ret_val, r_vec, t_vec = cv2.solvePnP(object_points, imagePoints, camera_matrix, dist_coefs)


    # ++++++ draw object with r_vec and t_vec on top of rgb frame
    # We saw how to draw cubes in camera calibration. (copy paste)
    # after this works you can replace this with the draw function from the renderer class renderer.draw() (1 line)
    drawn_image = renderer.draw(frame_rgb, r_vec, t_vec)
    drawn_image_bgr = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)


    # =========== plot and save frame
    video_out.write(drawn_image_bgr)
    ret, frame = cap.read()
# ======== end all
cap.release()
video_out.release()
