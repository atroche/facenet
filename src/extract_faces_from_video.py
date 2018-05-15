import os
from typing import List

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# running this more than once in same console throws an exception, hence this guard:
try:
    detector = MTCNN()
except ValueError as ve:
    # doing the job of this silly library:
    if not ve.args[0].startswith("Variable pnet/conv1/weights already exists"):
        raise ve

PADDING = 32


def cropped_face(image: np.ndarray, bounding_box: List[int]) -> np.ndarray:
    x, y, w, h = bounding_box
    top = max(0, y - h)
    bottom = y + 2 * h
    left = max(0, x - w)
    right = x + 2 * w
    return image[top:bottom, left:right]
#
#
#
# img_array = cv2.imread("datasets/videos/manual_zuck_screenshot.png")
# faces = detector.detect_faces(img_array)
# cv2.imwrite("datasets/videos/uncropped_face.jpg", img_array)
# cv2.imwrite("datasets/videos/cropped_face.jpg", cropped_face(img_array, faces[0]['box']))



videos_dir = "datasets/videos/"
filename = "how_to_build_the_future.mkv"

video_path = videos_dir + filename
filename_w_o_extension = filename[:-4]
faces_dir = "datasets/videos/faces/%s_better_crop/MZ/" % filename_w_o_extension
os.makedirs(faces_dir, exist_ok=True)

capture = cv2.VideoCapture(video_path)

while capture.isOpened():
    frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES)

    ret, frame = capture.read()

    # proxy for end of video:
    if not ret:
        break

    # only detect faces every FPS frames (i.e. every second):
    if not (0 == int(frame_number % capture.get(cv2.CAP_PROP_FPS))):
        continue

    # just for testing purposes, don't want to do entire video
    # if frame_number > 30 * 10:
    #     break

    faces = detector.detect_faces(frame)

    for face_num, face in enumerate(faces):
        cropped = cropped_face(frame, face['box'])
        filename = faces_dir + "frame_%s_face_%s.jpg" % (frame_number, face_num)
        cv2.imwrite(filename, cropped)

# releasing for now, because it seems to prevent me from opening the file in VLC (?!)
capture.release()


