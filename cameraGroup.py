
import cv2
import numpy as np

#     上：3
#左：1 中：0 右：2(面向摄像头)
#     下：4

class CameraParam:
    def __init__(self, id: int, R: cv2.typing.MatLike,K:cv2.typing.MatLike, direction: str):
        self.id = id
        self.R =R
        self.K = K
        #TODO:也许用字典更好
        match self.id:
            case 0:
                self.RotateAngle = -1
            case 1:
                self.RotateAngle = cv2.ROTATE_90_CLOCKWISE
            case 2:
                self.RotateAngle = cv2.ROTATE_90_COUNTERCLOCKWISE
            case 3:
                self.RotateAngle = cv2.ROTATE_180
            case 4:
                self.RotateAngle = -1
            case _:
                self.RotateAngle = -1

        self.direction = direction
        self.cap = cv2.VideoCapture(id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def is_open(self):
        return self.cap.isOpened()

    def read_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


def init():
    cameras = []
    #面向摄影机z轴负向
    direction = ["center", "left", "right",  "top","bottom"]

    print("正在初始化摄像头...")
    for i in range(5):
        # 尝试创建摄像头对象
        try:
            K = np.load(r'/\cameraParams\K.npy')
        except FileNotFoundError:
            print(f"错误：未找到内参矩阵文件K.npy，请确定当前正在进行相机校准")
            K = np.eye(3, k=1, dtype=np.float32)
        if i == 0:
            R = cv2.Rodrigues(np.array([[0], [0], [0]], np.float32))[0]
        else:
            try:
                R = np.load(r'C:\Users\mrblue\PycharmProjects\DoubleFishEyeCam\cameraParams\R-' + direction[i] + '.npy')
            except FileNotFoundError:
                print(f"警告: 未找到摄像头 {i} 的旋转矩阵文件 R-{direction[i]}.npy，使用默认值")
                R = np.eye(3, k=1, dtype=np.float32)

        # 创建摄像头参数对象
        cam_param = CameraParam(i, R, K,direction[i])

        # 检查摄像头是否成功打开
        if cam_param.is_open():
            cameras.append(cam_param)
            print(f"摄像头 {cam_param.id} 初始化成功")
        else:
            cameras.append(None)
            print(f"摄像头 {cam_param.id} 未接入或无法打开，已忽略")
            cam_param.release()  # 释放未能打开的摄像头资源

    return cameras
