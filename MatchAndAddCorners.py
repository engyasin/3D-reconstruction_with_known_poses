"""
a class for matching using optical-flow
and putting the result in pointStruct class

"""
import numpy as np
import cv2
from pointStruct import pointStruct


class MatchAndAddCorners(self):
    def __init__(self):
        self.matchBackThresh = 1
        self.feature_params = dict( maxCorners = 1500,
                                    qualityLevel = 0.3,
                                    minDistance = 7,
                                    blockSize = 7 )
        self.lk_params = dict( winSize  = (15, 15),
                               maxLevel = 2,
                               criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    def Match(self,I0,p0,I1):
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(I0, I1, p0, None, **self.lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(I1, I0, p1, None, **self.lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < self.matchBackThresh
        
        return good,p1
    
    def AddCorners(self,I1):
        mask = np.zeros_like(I1)
        mask[:] = 255
        p = cv2.goodFeaturesToTrack(I1, mask = mask, **self.feature_params)
        d = []
        for e in p:
            d.append(np.all(map(abs,self.p1r-e)>self.closeToCorner))
        p1r = [x for x,y in zip(p,d) if y]
        return p1r
    
    def Advance(self,I0,p0,I1):
        good,p1 = self.Match(I0,p0,I1)
        p1r = self.AddCorners(I1)
        kps = np.array([x[0] for x in kp])
        return good,kps,p1
    

