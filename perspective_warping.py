# ======= imports
import cv2
import numpy as np


# === template image keypoint and descriptors
t_im = cv2.imread("template_img.jpg")
t_im_gray = cv2.cvtColor(t_im, cv2.COLOR_BGR2GRAY)

wrapping_im = cv2.imread("py_in_forest.png")

dim = (t_im.shape[1], t_im.shape[0])
wrapping_im_resized = cv2.resize(wrapping_im, dim, interpolation = cv2.INTER_AREA)

feature_extractor = cv2.SIFT_create()

t_kp, t_desc = feature_extractor.detectAndCompute(t_im_gray, None)

# ===== video input, output and metadata
video_name = "input_video.mp4"
cap = cv2.VideoCapture(video_name)
ret, frame = cap.read()
height, width, layers = frame.shape
frameSize=(width, height)
fps = cap.get(cv2.CAP_PROP_FPS)
video_out = cv2.VideoWriter('output_video_part1.avi',fourcc=cv2.VideoWriter_fourcc(*'DIVX'),fps=fps, frameSize=frameSize)


# ========== run on all frames
while(ret): #while True: 
    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f_kp, f_desc = feature_extractor.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(f_desc, t_desc, k=2)

    good_f_kp = np.array([f_kp[m.queryIdx].pt for m in np.asarray(matches)[:,0]])
    good_t_kp = np.array([t_kp[m.trainIdx].pt for m in np.asarray(matches)[:,0]])

    # ======== find homography
    # also in SIFT notebook
    H, masked = cv2.findHomography(good_t_kp, good_f_kp, cv2.RANSAC, 5.0)

    # ++++++++ do warping of another image on template image
    warped = cv2.warpPerspective(wrapping_im_resized, H, (frame.shape[1], frame.shape[0]))
    res_frame = np.where(warped == 0, frame,  warped)


    # =========== plot and save frame
    video_out.write(res_frame)
    ret, frame = cap.read()

# ======== end all
cap.release()
video_out.release()
