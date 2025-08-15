

import cv2
import numpy as np

import calibrator

Dict={"left":"Left","right":"Right","top":"Top","bottom":"Bottom"}




direction=input("输入标定摄像机的方位:\n")
pathA=r'C:\Users\mrblue\PycharmProjects\DoubleFishEyeCam\caliPictures\calibrationPhotos'+Dict[direction]+r'\A'
pathB=r'C:\Users\mrblue\PycharmProjects\DoubleFishEyeCam\caliPictures\calibrationPhotos'+Dict[direction]+r'\B'

cal=calibrator.Calibrator(8,7,(1280,720))

objpA,imgpA=cal.getConnerPoints(pathA,False)
objpB,imgpB=cal.getConnerPoints(pathB,False)

_,cameraMtxA,disCoeffsA,rvecA,tvecA=cal.oneEyeCalibrate(objpA,imgpA)
_,cameraMtxB,disCoeffsB,rvecB,tvecB=cal.oneEyeCalibrate(objpB,imgpB)

retS,_,_,_,_,R,T,_,_=cv2.stereoCalibrate(objpA,imgpA,imgpB,cameraMtxA,disCoeffsA,cameraMtxB,disCoeffsB,(1280,720),flags=cv2.CALIB_FIX_INTRINSIC,criteria=
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001,))

print("cameraMtxA: ",cameraMtxA)
print("reprojS: ",retS)
print("R: ",R)
print("T: ",T)

rvec_A2B,_=cv2.Rodrigues(R.T)

np.save("cameraParams/K",cameraMtxA.astype(np.float32))
np.save("cameraParams/R-"+direction,R.astype(np.float32))
np.save("cameraParams/T-"+direction,T)

print("rvec_B2A: ",rvec_A2B)
