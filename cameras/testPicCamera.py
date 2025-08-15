### 这段程序用于拍摄一组用于检验效果的照片
import cv2
import cameraGroup

# 初始化摄像头组
cameras = cameraGroup.init()
directMap={"left":1,"top":2,"right":3,"bottom":4}
# 确保摄像头初始化成功
if cameras is None:
    print("摄像头初始化失败，程序退出")
    exit()

while True:
    frames = []
    for camera in cameras:
        if camera is None:
            continue
        ret, frame = camera.read_frame()
        if not ret:
            continue
        # 根据摄像头方向旋转帧
        if camera.RotateAngle != -1:
            frame = cv2.rotate(frame, camera.RotateAngle)
        frames.append((camera.id, frame))
        cv2.imshow(f'frame{camera.id}', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC键退出
        break
    elif k == ord('s'):  # 's'键保存照片
        for index, frame in frames:
            filename = f"../testPictures/img_{cameras[index].direction}.jpg"
            if cv2.imwrite(filename, frame):
                print(f'照片已保存: {filename}')
            else:
                print(f'保存照片失败: {filename}')

# 释放摄像头资源
for camera in cameras:
    if camera is None:
        continue
    camera.release()

cv2.destroyAllWindows()