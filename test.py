import torch
import cv2

def test_yolov5_face():
    model = torch.hub.load('/Users/xuekejun/PycharmProjects/yolov5', 'custom', path='weights/yolov5s.pt',
                           source='local').eval()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
        frame = cv2.flip(frame, 1)
        results = model(frame)
        print(results.pandas().xyxy[0]['name'].values.tolist())
        c = cv2.waitKey(50)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_yolov5_face()