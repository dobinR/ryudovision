import cv2

video_name = 'p1.mp4'
cap = cv2.VideoCapture(video_name)

while True:
    ret, src1 = cap.read()
    if not ret:
        break 
    cv2.imshow('src', src1)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break 