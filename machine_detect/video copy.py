import cv2 
import detector
import time 
import timeit 

video_name = 'video/leg_extension1.mp4'
video = cv2.VideoCapture(video_name)

prev_time = 0
FPS = 2 

while True:
    ret, frame = video.read()
    
    if ret is True:
        start_t = timeit.default_timer()
        terminate_t = timeit.default_timer()
        FPS = int(1./(terminate_t - start_t))
        cv2.imshow('video', frame)
        print(FPS)
        if cv2.waitKey(1) > 0:
            break