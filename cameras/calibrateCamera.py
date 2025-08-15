### 这段程序用于拍摄一组用于标定的照片
import cv2
import os

import cameraGroup

#     上：3
#左：1 中：0 右：2(面向摄像头)
#     下：4

direction=input("输入标定摄像头方向\n")
print(direction)
directMap={"left":1,"top":3,"right":2,"bottom":4}


# 初始化摄像头组
cameras = cameraGroup.init()

# 确保摄像头初始化成功
if cameras is None:
    print("摄像头初始化失败，程序退出")
    exit()

# 创建保存标定照片的目录
calibration_dirs = {
    "right": "../caliPictures/calibrationPhotosRight",
    "bottom": "../caliPictures/calibrationPhotosBottom",
    "left": "../caliPictures/calibrationPhotosLeft",
    "top": "../caliPictures/calibrationPhotosTop"
}

for directory in calibration_dirs.values():
    if not os.path.exists(directory):
        os.makedirs(directory)

picNum = 0

while True:
    frames = []
    _,frameA=cameras[0].read_frame()
    _,frameB=cameras[directMap[direction]].read_frame()

    if cameras[directMap[direction]].RotateAngle!=-1:
        frameB=cv2.rotate(frameB,cameras[directMap[direction]].RotateAngle)

    cv2.imshow("frameA", frameA)
    cv2.imshow("frameB", frameB)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC键退出
        break
    elif k == ord('s'):  # 's'键保存照片
        success = True
        cv2.imwrite(f'{calibration_dirs[direction]}/A/img{picNum}.jpg', frameA)
        cv2.imwrite(f'{calibration_dirs[direction]}/B/img{picNum}.jpg', frameB)
        if success:
            print(f'pic{picNum} captured!')
            picNum += 1

# 释放摄像头资源
for camera in cameras:
    if camera is not None:
        camera.release()

cv2.destroyAllWindows()