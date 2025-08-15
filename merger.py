import math

import numpy as np


class Merger:
    def __init__(self):
        self.masks = None

    def generateMask(self, images, centers,sz):
        masks = [np.zeros((sz[0], sz[1], 3), np.float32) for _ in range(5)]
        for i in range(sz[0]):
            for j in range(sz[1]):
                distance = []
                for index in range(len(images)):
                    if images[index][i][j][0] == 0 and images[index][i][j][1] == 0 and images[index][i][j][2] == 0:
                        distance.append(-1)
                    else:
                        #采用倒数加权，距离越大mask越淡
                        distance.append(1/(getDistance(centers[index], j, i)+1e-10))
                for index in range(len(images)):
                    sum=0
                    for dist in distance:
                        if dist == -1:
                            continue
                        sum += dist
                    if distance[index] == -1:
                        masks[index][i][j][0] = 0
                        masks[index][i][j][1] = 0
                        masks[index][i][j][2] = 0
                    else:
                        masks[index][i][j][0] = distance[index] / sum
                        masks[index][i][j][1] = distance[index] / sum
                        masks[index][i][j][2] = distance[index] / sum
        self.masks = masks
    def merge(self,images,centers,canva_sz):
        if self.masks is None:
            self.generateMask(images,centers,canva_sz)

        res = np.zeros((canva_sz[0], canva_sz[1], 3), np.float32)
        for i in range(len(images)):
            res += self.masks[i] * images[i] / 255
        return res

def getDistance(center, x, y):
    return math.sqrt((center[0] - x) ** 2 + (center[1] - y) ** 2)
