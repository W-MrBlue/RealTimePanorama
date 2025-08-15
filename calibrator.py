import glob

import cv2
import numpy as np


class Calibrator:
    def __init__(self, w: int, h: int, imgSize: tuple[int, int]):
        self.w = w
        self.h = h
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001,)
        self.imgSize = (imgSize[0], imgSize[1])

    def getConnerPoints(self, path: str, showConners=False):

        objp = np.zeros((self.w * self.h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.w, 0:self.h].T.reshape(-1, 2)
        objp = objp * 35

        objPoints = []
        imgPoints = []
        connerImg = None

        folder = glob.glob(path + '/*.jpg')
        if len(folder) == 0:
            print('No images found')

        for file in folder:
            img = cv2.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.w, self.h), None)
            if not ret or len(corners) != 56:
                print('pic: ' + file + ' Failed to find corners')
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.cornerSubPix(gray, corners, (6, 6), (-1, -1), self.criteria)
                objPoints.append(objp)
                imgPoints.append(corners)
                connerImg = cv2.drawChessboardCorners(None, (self.w, self.h), corners, ret)

        if not showConners:
            return objPoints, imgPoints
        else:
            return objPoints, imgPoints, connerImg

    def oneEyeCalibrate(self, objPoints, imgPoints):
        reproj, cameraMtx, disCoeffs, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, self.imgSize,
                                                                         None, None, None, None,
                                                                         None, self.criteria)
        return reproj, cameraMtx, disCoeffs, rvecs, tvecs

    def doubleEyeCalibrate(self, objPoints, imgPointsA, imgPointsB, cameraMtxA, disCoeffsA, cameraMtxB, disCoeffsB):
        reproj, cameraMtxA, disCoeffsA, cmaeraMtxB, discoeffsB, R, T, E, F = cv2.stereoCalibrate(objPoints, imgPointsA,
                                                                                                 imgPointsB,
                                                                                                 cameraMtxA, disCoeffsA,
                                                                                                 cameraMtxB, disCoeffsB,
                                                                                                 self.imgSize,
                                                                                                 flags=cv2.CALIB_FIX_INTRINSIC,
                                                                                                 criteria=self.criteria)
        return reproj, cameraMtxA, disCoeffsA, cmaeraMtxB, discoeffsB, R, T, E, F

    def rtCaculator(self, pathA, pathB):
        objPointsA, imgPointsA = self.getConnerPoints(pathA, False)
        objPointsB, imgPointsB = self.getConnerPoints(pathB, False)

        retA, cameraMtxA, disCoeffsA, rvecA, tvecA = self.oneEyeCalibrate(objPointsA, imgPointsA)
        retB, cameraMtxB, disCoeffsB, rvecB, tvecB = self.oneEyeCalibrate(objPointsB, imgPointsB)

        retS, _, _, _, _, R, T, _, _ = self.doubleEyeCalibrate(objPointsA, imgPointsA, imgPointsB,
                                                               cameraMtxA, disCoeffsA, cameraMtxB, disCoeffsB)
        return retA, retB, retS,cameraMtxA, disCoeffsA,cameraMtxB, disCoeffsB, R,T

    def getMaps(self, R, T, cameraMtxA, disCoeffsA, cameraMtxB, disCoeffsB,alpha:float):
        rotA, rotB, projA, projB, _, _, _ = cv2.stereoRectify(cameraMtxA, disCoeffsA, cameraMtxB, disCoeffsB,
                                                              self.imgSize,
                                                              R, T, None, None, None, None, None,
                                                              cv2.CALIB_ZERO_DISPARITY, alpha)
        map1A, map2A = cv2.initUndistortRectifyMap(cameraMtxA, disCoeffsA, rotA, projA, self.imgSize, cv2.CV_32FC1)
        map1B, map2B = cv2.initUndistortRectifyMap(cameraMtxB, disCoeffsB, rotB, projB, self.imgSize, cv2.CV_32FC1)
        return map1A, map2A, map1B, map2B
