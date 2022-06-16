#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import picamera
import numpy as np
import dlib
import math

# This code uses the 5 point face landmarking model, which can be downloaded from
# http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2


# start the Pi camera in sensor mode 2 - capturing the full frame
with picamera.PiCamera(framerate=15, sensor_mode=2) as camera:
    camera.vflip = True                             # I had to mount the camera module upside down
    camera.resolution = (2592, 1944)                # full sensor resolution
    preview = camera.start_preview()                # start previewing the video in fullscreen

    cvw = 320                                       # run the face detection at a lower resolution
    cvh = int(cvw * camera.resolution[1] / camera.resolution[0])
    cvscale = float(camera.resolution[0]) / float(cvw)
    bufsize = int(cvw * cvh * 3 / 2)
    yuv = np.empty((bufsize,), dtype=np.uint8)

    screenaspect = 576.0 / 720.0                    # PAL aspect ratio

    import cv2 as cv
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

    print("starting")
    for frame in camera.capture_continuous(yuv, format="yuv", resize=(cvw, cvh), use_video_port=True):
        image = yuv[:cvw*cvh].reshape((cvh, cvw))

        faces = detect(image, 0)                    # detect faces in the camera image

        if len(faces) > 0:                          # if any faces are present...
            landmarks = predict(image, faces[0])    # ...fit the 5 face landmarks to the first one

            eyecorners = (landmarks.part(2) * cvscale, landmarks.part(3) * cvscale)
            eyecentre = (eyecorners[0] + eyecorners[1]) * 0.5
            eyeradius = dlib.length(eyecorners[1] - eyecorners[0]) * 0.35
            eyebox = (eyeradius, eyeradius * screenaspect)

            x = min(max(eyebox[0], eyecentre.x), camera.resolution[0] - eyebox[0])
            y = min(max(eyebox[1], eyecentre.y), camera.resolution[1] - eyebox[1])
            
            # update the preview to use a cropped portion of the video around the eye we detected
            preview.crop = (int(x - eyebox[0]), int(y - eyebox[1]), int(eyebox[0] * 2), int(eyebox[1] * 2))


