import torch
import numpy as np
import cv2
from time import time

class ObjectDetection:
    # YouTube 동영상에 YOLOv5 구현
    def __init__(self, name, out_file):
        # 객체 생성 시 호출
        # url: 예측 대상 YouTube URL
        # out_file: 유효한 출력 파일 이름 *.avi
        self.video_name = name
        self.model = self.load_model()
        self.classes = {0:'hip_truster', 1:'leg_extension', 2:'leg_press', 3:'lying_leg_curl', 4:'v_squat', 5:'hack_squat'}
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def get_video(self):
        # url에서 새 비디오 스트리밍 객체 생성
        return cv2.VideoCapture(self.video_name)
    def load_model(self):
        # YOLOv5 모델 로드
        model = 'weights/best_robo250'
        return model
    def score_frame(self, frame):
        # frame: 단일 프레임; numpy/list/tuple 형식
        # return: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord
    def class_to_label(self, x):
        # x 숫자 레이블 -> 문자열 레이블로 반환
        return self.classes[int(x)]
    def plot_boxes(self, results, frame):
        # 경계상자와 레이블을 프레임에 플로팅
        # results: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        # frame: 점수화된 프레임
        # return: 경계 상자와 레이블이 플로팅된 프레임
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i])
                            + ': ' + str(x1) + ', ' + str(x2) + ', ' + str(y1) + ', ' + str(y2),
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame
    def __call__(self):
        # 인스턴스 생성 시 호출; 프레임 단위로 비디오 로드
        player = self.get_video()
        # assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        while True:
            start_time = time()
            ret, frame = player.read()
            # assert ret
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")
            out.write(frame)


Video = ObjectDetection("leg_extension1", "Black_Bear_YOLOv5.avi")
Video()