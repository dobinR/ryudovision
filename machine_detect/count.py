import cv2 
import detector
import time 

video_name = 'v_squat.mp4'
cap = cv2.VideoCapture(video_name)

prev_time = 0
FPS = 6

machine_cnt = {'hip_truster':0, 'leg_extension':0, 'leg_press':0, 
                'lying_leg_curl':0, 'v_squat':0, 'hack_squat':0}

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
            machine, prob = detector.cnt_cls(det)
            if float(prob) >= 0.8:
                machine_cnt[machine] += 1
                print(machine_cnt)
        else:
            dst = src.copy()
        
        cv2.namedWindow('detect', flags=cv2.WINDOW_NORMAL)

        cv2.imshow('detect', dst)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print(max(machine_cnt, key=machine_cnt.get))
            break 

cap.release()