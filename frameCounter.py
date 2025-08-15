import time


class FrameCounter:
    def __init__(self):
        self.frameCnt=0
        self.timer=time.time()

    def countFps(self)->float:
        self.frameCnt+=1
        if self.frameCnt==10:
            fps=self.frameCnt/(time.time()-self.timer)
            self.timer=time.time()
            self.frameCnt=0
            return fps
        return -1.0