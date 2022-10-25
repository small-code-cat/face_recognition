# face_recognition

#### 介绍
人脸检测+活体检测+人脸识别+徘徊检测

#### 运行环境
python >= 3.7
其他的包缺什么补什么即可

#### 使用说明
1.在main.py中首先运行get_face()函数
2.将get_face()那行代码删除，再运行main函数即可，传的参数是代表你想检测多少秒的视频

注意：本项目中默认陌生人出镜率超过50%就算是徘徊了，这个参数可以在main.py中的get_VideoTracker中修改--linger_thres，默认是0.5

