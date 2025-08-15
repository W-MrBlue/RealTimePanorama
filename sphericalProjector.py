import stitcher
import cv2
import numpy as np

from merger import Merger


class SphericalProjector:
    def __init__(self):
        pass

    #传入K,R,返回roi和imgWp
    def simpleProject(self, src: cv2.typing.MatLike, K: cv2.typing.MatLike,
                      R: cv2.typing.MatLike, f: float) -> tuple[cv2.typing.Point, cv2.typing.MatLike]:
        warper = cv2.PyRotationWarper('spherical', f)
        conner, res = warper.warp(src, K, R, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        return conner, res

    def trans2canva(self, src: cv2.typing.MatLike, canva_sz: cv2.typing.Point,
                    conner: cv2.typing.Point) -> (tuple[int,int],cv2.typing.MatLike):
        canva = np.zeros((canva_sz[1], canva_sz[0], 3), np.uint8)
        x, y = conner
        x += canva.shape[1] // 2
        #print(x,y,src.shape[:-1])
        canva[y:y + src.shape[0], x:x + src.shape[1]] = src
        roi_center=(x+(src.shape[1]//2),y+(src.shape[0]//2))
        #debug
        #canva=cv2.rectangle(canva,(x,y),(x + src.shape[1],y + src.shape[0]),(0,255,0),2)
        return roi_center,canva

    def project(self, src: cv2.typing.MatLike,canva_sz: cv2.typing.Point, K: cv2.typing.MatLike, R: cv2.typing.MatLike, f: float):
        conner, dst = self.simpleProject(src, K, R, f)
        return self.trans2canva(dst, canva_sz, conner)


if __name__ == '__main__':
    direction = input("输入投影摄像机方位:\n")
    frameA = cv2.imread('testPictures/img_center.jpg')
    frameB = cv2.imread('testPictures/img_' + direction + '.jpg')
    K = np.load('cameraParams/K.npy')
    R0 = np.array([[0], [0], [0]], np.float32)
    R1 = np.load('cameraParams/R-' + direction + '.npy')
    print(K, R1)

    canvaSize = (1920, 1080)
    focal = K[1, 2]
    print(focal)
    #投影器初始化
    projector = SphericalProjector()

    #接合器初始化
    stitcher = stitcher.Stitcher()

    #球面投影
    connerA, dstA = projector.simpleProject(frameA, K, cv2.Rodrigues(R0)[0], focal)
    centerA,dstA = projector.trans2canva(dstA, canvaSize,connerA)
    connerB, dstB = projector.simpleProject(frameB, K, R1.T, focal)
    centerB,dstB = projector.trans2canva(dstB, canvaSize, connerB)

    #特征匹配
    match,centerB,result = stitcher.stitch((dstA, dstB), centerB,direction, showMatches=True)
    merger=Merger()
    masks=merger.generateMask((dstA,result),(centerA,centerB),dstA.shape[:2])
    cv2.imshow('matches',match)
    cv2.imshow('contrast', cv2.addWeighted(dstA, 0.5,dstB, 0.5, 0))
    cv2.imshow('stitched', masks[0]*dstA/255+masks[1]*result/255)

    #cv2.imshow('zoom_in', cv2.resize(cv2.addWeighted(dstA, 0.5, result, 0.5, 0)[260:720, 600:1300], (0, 0), fx=2, fy=2, interpolation=cv2.INTER_AREA))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
