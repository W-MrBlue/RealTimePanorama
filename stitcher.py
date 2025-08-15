import time
from pickletools import uint8

import cv2
import numpy as np


class Stitcher:
    def __init__(self):
        self.TM = None
        self.RM = None
        self.BM = None
        self.LM = None
        self.DictM={"left":self.LM, "right":self.RM, "top":self.TM, "bottom":self.BM}


    def stitch(self, images,center, direction,ratio=0.75, reprojThresh=4.0,
               showMatches=False):
        # unpack the images, then detect keypoints and extract local invariant descriptors from them
        #@imageA,@imageB and @overlapMask are of the same size
        (imageB, imageA) = images
        if self.DictM[direction] is None:
            print("caculate M of "+direction)
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)
            # match features between the two images
            self.DictM[direction]= self.matchKeypoints(kpsA, kpsB,
                                                       featuresA, featuresB, ratio, reprojThresh)
            # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
            if self.DictM[direction] is None:
                return None

        # otherwise, apply a perspective warp to stitch the images together
        (matches, H, status) = self.DictM[direction]

        center_homogeneous = np.array([center[0], center[1], 1])
        new_point = H @ center_homogeneous  # 矩阵乘法
        new_point_center = (
            int(new_point[0] / new_point[2]),  # x 坐标
            int(new_point[1] / new_point[2])  # y 坐标
        )

        result = cv2.warpPerspective(imageA, H,(imageA.shape[1], imageA.shape[0]))

    # check to see if the keypoint matches should be visualized
        if showMatches:
            start = time.time()
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)
            end = time.time()
            print('%.5f s' % (end - start))
            # return a tuple of the stitched image and the
            # visualization
            return vis,new_point_center,result
        else:
            return new_point_center, result
    # 接收照片，检测关键点和提取局部不变特征
    # 用到了高斯差分（Difference of Gaussian (DoG)）关键点检测，和SIFT特征提取
    # detectAndCompute方法用来处理提取关键点和特征
    # 返回一系列的关键点
    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect keypoints in the image
        detector = cv2.SIFT.create()
        kps,features = detector.detectAndCompute(gray, None)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    # matchKeypoints方法需要四个参数，第一张图片的关键点和特征向量，第二张图片的关键点特征向量。
    # David Lowe’s ratio测试变量和RANSAC重投影门限也应该被提供。
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.BFMatcher()
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        #debug
        #print(len(rawMatches))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            H, status= cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    # 连线画出两幅图的匹配
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return cv2.resize(vis, None, fx=0.5, fy=0.5)