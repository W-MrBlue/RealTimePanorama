import cv2
import numpy as np

import cameraGroup
import stitcher
import frameCounter
from sphericalProjector import SphericalProjector
from merger import Merger

directMap = {"left": 1, "top": 2, "right": 3, "bottom": 4}
# 投影器初始化
projector = SphericalProjector()

# 接合器初始化
stitcher = stitcher.Stitcher()

# 帧率统计初始化
fpsCounter = frameCounter.FrameCounter()
fps = -1.0

#摄像头组初始化
cameras = cameraGroup.init()

#融合器初始化
merger = Merger()

# 从第一个摄像头获取K矩阵
K = cameras[0].K
canvaSize = (1920, 1080)
focal = K[1, 2]

firstRoundFlag = True
while True:
    # 读取所有摄像头的帧
    frames = []
    centers=[(0,0) for _ in range(5)]
    if firstRoundFlag:
        # 使用测试图片
        for cam in cameras:
            if cam is None:
                frames.append(None)
                continue
            frame = cv2.imread(
                r'C:\Users\mrblue\PycharmProjects\DoubleFishEyeCam\testPictures\img_' + cam.direction + '.jpg')
            frames.append(frame)
            firstRoundFlag = False
    else:
        # 从实际摄像头读取
        for cam in cameras:
            if cam is None:
                frames.append(None)
                continue
            ret, frame = cam.read_frame()
            if ret:
                if cam.RotateAngle != -1:
                    frame = cv2.rotate(frame, cam.RotateAngle)
                frames.append(frame)
            else:
                frames.append(None)

    mergeFrames=[]
    mergeCenters=[]
    # 处理帧（这里以两个摄像头为例，实际应用中可以根据需要扩展）
    for i in range(5):
        if frames[i] is not None:
            if i==0:
                centers[i],frames[i] = projector.project(frames[i],canvaSize, K, cameras[i].R.T, focal)
            else:
                centers[i], frames[i] = projector.project(frames[i], canvaSize, K, cameras[i].R.T, focal)
                centers[i],frames[i] = stitcher.stitch((frames[0], frames[i]), centers[i],cameras[i].direction, showMatches=False)
            mergeFrames.append(frames[i])
            mergeCenters.append(centers[i])


    cv2.imshow('all',merger.merge(mergeFrames,mergeCenters,mergeFrames[0].shape[:2]))

    # 帧率统计
    getFps = fpsCounter.countFps()
    if getFps != -1.0:
        fps = getFps

    print("current fps:", fps)
    # 帧率显示

    # 检查退出按键
    if cv2.waitKey(1) == 27:
        # 释放所有摄像头资源
        for cam in cameras:
            if cam is None:
                continue
            cam.release()
        cv2.destroyAllWindows()
        break
