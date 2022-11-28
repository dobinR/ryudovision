import cv2 
import detector
import time 

video_name = 'v_squat.mp4'
cap = cv2.VideoCapture(video_name)

prev_time = 0
FPS = 6

while True:
    ret, src = cap.read()
    # src = cv2.flip(src, 0)
    current_time = time.time() - prev_time 
    if not ret:
        break
    elif (ret is True) and (current_time > 1./ FPS) :
        prev_time = time.time()
        det = detector.detect(src)
        if det is not None:
            dst = detector.draw_boxes(src, det)
        else:
            dst = src.copy()
        
        cv2.namedWindow('detect', flags=cv2.WINDOW_NORMAL)

        cv2.imshow('detect', dst)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break 

cap.release()

# while True:
#     ret, src = cap.read()
#     if not ret:
#         break 
#     det = detector.detect(src)
#     if det is not None:
#         dst = detector.draw_boxes(src, det)
#     else:
#         dst = src.copy()
#     cv2.imshow('dst', dst)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break 