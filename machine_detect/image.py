import cv2
from detector import detect, draw_boxes

image_name = 'machine_detect/image/레그프레스 머신91.jpg'
src = cv2.imread(image_name)
det = detect(src)

if det is not None:
    dst = draw_boxes(src, det)
    cv2.imshow('detection', dst)
    cv2.waitKey(0)
else:
    print('no object found')