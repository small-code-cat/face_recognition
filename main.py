import shutil

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import argparse
from linger_detection.VideoTracker import VideoTracker
from linger_detection.yolov5.utils.general import check_img_size
from my_utils import plot_one_box
import time
import pyautogui
from sklearn.metrics.pairwise import cosine_similarity
from silentFace_model.predict_net import *
from silentFace_model.generate_patches import CropImage

def get_face():
    root_dir = 'face_imgs'
    img_list = []
    cap = cv2.VideoCapture(0)  # 0为电脑内置摄像头
    words = {
        0:'请正面面对摄像头',
        1:'请脸向左转45度',
        2:'请脸向右转45度',
        3:'请抬头45度',
        4:'请低头45度'
    }
    i = 0
    while (True):
        pyautogui.alert(words[i]+'(点击ok即可开始拍照)')
        ret, frame = cap.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
        frame = cv2.flip(frame, 1)
        img_list.append(frame)
        i += 1
        if i==5:
            break
    cap.release()
    cv2.destroyAllWindows()
    while True:
        name = pyautogui.prompt('请输入你的名字拼音缩写(eg:张三->zs)')
        if name=='' or name is None:
            continue
        break
    save_path = root_dir+os.sep+name
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    for i,frame in enumerate(img_list):
        cv2.imwrite(save_path+os.sep+str(i)+'.jpg', frame)
    make_dataset()

def make_dataset():
    img_path = 'face_imgs'
    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    d = {}
    for cls in os.listdir(img_path):
        if cls.startswith('.'):
            continue
        work_path = img_path+os.sep+cls
        if cls not in d:
            d[cls] = []
        for i in os.listdir(work_path):
            img = Image.open(work_path+os.sep+i)
            img_cropped = mtcnn(img, save_path='output'+os.sep+cls+os.sep+i)
            img_embedding = resnet(img_cropped.unsqueeze(0))
            d[cls].append(img_embedding)
        d[cls] = torch.stack(d[cls]).squeeze(dim=1)
    torch.save(d, 'database.pth')

def face_detection(mtcnn, img):
    '''
    人脸检测
    :param mtcnn: mtcnn模型
    :param img_np: numpy类型
    :return: 人脸特征，人脸的xyxy
    '''
    # Get cropped and prewhitened image tensor
    img_cropped, xyxy = mtcnn(img, return_prob=True)
    xyxy = xyxy.squeeze(axis=0).tolist() if xyxy is not None else xyxy
    return (img_cropped, xyxy)

def live_detection(anti_model, image_cropper, frame, xyxy):
    prediction = np.zeros((1, 3))
    silentFace_weights = 'weights/anti_spoof_models'
    for model_name in os.listdir(silentFace_weights):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": xyxy,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        rf_img = image_cropper.crop(**param)
        prediction += anti_model.predict(rf_img, os.path.join(silentFace_weights, model_name))
    rf_label = np.argmax(prediction)
    value = prediction[0][rf_label] / 2
    is_live = (rf_label == 1 and value > 0.9)
    return is_live

def face_recognition(resnet, img_cropped, database, thres=0.7):
    '''
    人脸识别
    :param resnet: 人脸识别模型
    :param img_cropped: 人脸特征
    :param database: 数据库中的人脸特征
    :return: 识别的人名
    '''
    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0)).detach().numpy()
    # print(f'人脸对应的特征向量为：\n{img_embedding}')
    # print(f'匹配阈值：{thres}')
    label = 'stranger'
    for k in database:
        cos_simi = cosine_similarity(img_embedding, database[k].detach().numpy())
        max = np.max(cos_simi)
        # print(f'与{k}的人脸的余弦相似度为：{max}')
        if max > thres:
            label = k
            # print(f'由于余弦相似度大于匹配阈值，故匹配成功')
        # else:
        #     print(f'由于余弦相似度小于匹配阈值，故匹配失败')
    return label

def get_VideoTracker():
    parser = argparse.ArgumentParser()
    parser.add_argument('--linger_thres', type=float, default=0.5, help='linger threshold')
    # input and output
    parser.add_argument('--input_path', type=str, default='sample.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_path', type=str, default='linger_detection/output/', help='output folder')  # output folder
    parser.add_argument("--frame_interval", type=int, default=2)
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_txt', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # camera only
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")

    # YOLO-V5 parameters
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # deepsort parameters
    parser.add_argument("--config_deepsort", type=str, default="./linger_detection/configs/deep_sort.yaml")

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)
    return VideoTracker(args)

def main(linger_time, video_path=None):
    '''
    linger_time为None时不启动徘徊检测
    video_path不为None时启动摄像头检测
    :param linger_time: float
    :return:
    '''
    input_source = 0 if video_path is None else video_path
    cap = cv2.VideoCapture(input_source)  # 0为电脑内置摄像头
    database = torch.load('database.pth')
    # 由于人脸检测误差太大，所以先由yolo检测图中是否含有person
    yolo = torch.hub.load('/Users/xuekejun/PycharmProjects/yolov5', 'custom', path='weights/yolov5s.pt',
                           source='local').eval()
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN()
    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    anti_model = AntiSpoofPredict(0)
    image_cropper = CropImage()

    v1 = v2 = 0 #v1->活体检测计数；v2人脸比对检测计数

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
    fps = 30  # 帧数
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
    out = cv2.VideoWriter('sample.mp4', fourcc, fps, (width, height))  # 写入视频
    out.release()

    while (True):
        ret, frame = cap.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
        frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。

        if out.isOpened():
            out.write(frame)
            if time.time() - tmp_time > linger_time:
                print('--------开始对录制视频徘徊检测--------')
                out.release()
                # 检测录制好的视频
                with get_VideoTracker() as video_tracker:
                    video_tracker.run()
            continue

        yolo_results = yolo(frame)
        yolo_cls_list = yolo_results.pandas().xyxy[0]['name'].values.tolist()
        if 'person' not in yolo_cls_list:
            v1 = v2 = 0
            continue

        img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        # cv2.imshow("video", frame)
        img_cropped, xyxy = face_detection(mtcnn, img)
        has_face = (img_cropped is not None) # 有可能是人遮挡住了脸，也有可能纯粹是没有人

        if not has_face:
            continue
        # plot_one_box(xyxy, frame, label='face', line_thickness=3)
        # cv2.imshow("video1", frame)

        is_live = live_detection(anti_model, image_cropper, frame, xyxy)
        if is_live:
            # plot_one_box(xyxy, frame, label='live', line_thickness=3)
            # cv2.imwrite('/Users/xuekejun/Desktop/live.jpg', frame)
            v1 = 0
            name = face_recognition(resnet, img_cropped, database)
            is_pass = (name != 'stranger')
            if is_pass:
                v2 = 0
                print(f'当前来访人员为：{name}，不启动徘徊检测！')
                continue
            else:
                v2 += 1
                if v2 == 3:
                    v2 = 0
                    if linger_time is not None:
                        print(f'当前来访人员未知，启动徘徊检测！')
                        # 开启徘徊检测
                        out = cv2.VideoWriter('sample.mp4', fourcc, fps, (width, height))  # 写入视频
                        tmp_time = time.time()
        else:
            # plot_one_box(xyxy, frame, label='no_live', line_thickness=3)
            # cv2.imwrite('/Users/xuekejun/Desktop/no_live.jpg', frame)
            v1 += 1
            if v1 == 3:
                v1 = 0
                if linger_time is not None:
                    # 开启徘徊检测
                    out = cv2.VideoWriter('sample.mp4', fourcc, fps, (width, height))  # 写入视频
                    tmp_time = time.time()

        c = cv2.waitKey(50)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(5)
    # get_face()
    # make_dataset()