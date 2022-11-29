import cv2 
import detector
import time 
import timeit 

video_name = 'leg_extension1.mp4'
video = cv2.VideoCapture(video_name)

while True:
    ret, frame = video.read()
    

    if ret is True:
        start_t = timeit.default_timer()
        det = detector.detect(frame)
        if det is not None:
            dst = detector.draw_boxes(frame, det)
        else:
            dst = frame.copy()
        terminate_t = timeit.default_timer()
        cv2.imshow('video', dst)
        FPS = int(1./(terminate_t - start_t))
        print(FPS)

        if cv2.waitKey(1) > 0:
            break