# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:45:14 2018

@author: Eng-Yasin
"""

import glob
from pdb import set_trace as bp
from time import time as watch
from MatchAndAddCorners import MatchAndAddCorners
from scipy import spatial as sp

import matplotlib.pyplot as plt
from pointStruct import pointStruct
############### THE IMPORTS ########################
import numpy as np
cimport numpy as cnp
from mpl_toolkits.mplot3d import Axes3D

#from multiprocessing import process
import cv2
import pcl
#from read_poses_ekf_sim import GetMatrices
#or for ekf file, 
from read_poses_sim import GetMatrices
from selectROI import SelectROI
#####################################################
import cython
##########################################################################
#Methods with njit
##########################################################################

ctypedef cnp.float32_t FLOAT
ctypedef cnp.uint8_t UINT8

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list GiveGrid(int step_size,list rect):
    #cdef list[(int,int,int)] grid
    cdef unsigned int x,y
    grid = [(x, y, step_size) for y in xrange(rect[1], rect[3], step_size)
                                for x in xrange(rect[0], rect[2], step_size)]
    return grid
cpdef  cnp.ndarray[FLOAT, ndim=2] strip_out(cnp.ndarray[FLOAT, ndim=3] arrayT):
    cdef unsigned int a,b
    a,b = arrayT.shape[0], arrayT.shape[2]
    return  arrayT.reshape((a, b))
    #cdef double[:,:] x
    #res =  [x[0] for x in arrayT]
    #return res
cpdef list calc_errors(cnp.ndarray[FLOAT,ndim=2] pn1,cnp.ndarray[FLOAT,ndim=2] pnt1,
                 cnp.ndarray[FLOAT,ndim=2] pn2,cnp.ndarray[FLOAT,ndim=2] pnt2):
    cdef cnp.ndarray[FLOAT,ndim=2] o ,p 
    cdef cnp.ndarray[FLOAT,ndim=1] x
    cdef list a,b,errors
    cdef double u,v
    o = np.subtract(pn1,pnt1)**2
    a = []
    b = []
    for x in np.transpose(o):
        a.append(np.sqrt(x[0]+x[1]))
    #a = np.sqrt(map(lambda x: x[0]+x[1],o.T))
    p = np.subtract(pn2,pnt2)**2
    for x in np.transpose(p):
        b.append(np.sqrt(x[0]+x[1]))
    #b = np.sqrt(map(lambda x: x[0]+x[1],o.T))
    errors = [(u+v)/2.0 for u,v in zip(a,b)]
    #errors = np.array(a) + np.array(b)
    #errors = np.mean(np.vstack((a,b)),axis=0)
    return errors

cpdef list get_good_2(list good,list good3DPointsIndx):
    cdef bint y
    good_2 = [x for x,y in zip(good,good3DPointsIndx) if y]
    #good_2 = [p for i,p in zip(good3DPointsIndx,good) if i]
    return good_2

MakeKpsGrid = lambda x:cv2.KeyPoint(*x)
#Sort2MostVar= lambda x:np.var(x[1])
cdef float Sort2MostVar(x):
    #for ORB
    cdef cnp.ndarray[UINT8,ndim=1] x1
    x1 = x[1]
    return np.var(x1)
#ArrangeNames = lambda x: int(x[len(direct)+1:-4])

##########################################################################
#End of Methods
##########################################################################

cdef class ImagesWithPoses(object):
    cdef public list all_poses, all_vec_pose, Rs, Ts, image_names, Rects, nodesAndProjs
    cdef public list pointsCloud, PointsBankActive , PointsBankNonActive, CurrPrjErr
    cdef public unsigned int SiftStep, Image_Index , UpperLimitOfFeats
    cdef public double watch
    cdef public float lowe_ratio , ReprojectThresh
    cdef public str direct, FeauterType
    cdef public cnp.float64_t[:,:] Instrinc_Mat
    #may need to enter it as float
    cdef public cnp.float32_t[:] Destoration
    cdef public cnp.uint8_t[:,:,:]   I_last
    #cdef public unsigned char [:,:,:]   I_last
    cdef public object activeFeat, activeFlann , PointsFile , ProjsFile,OptFlow

    def __init__(self,int SiftStep,str direct="",int checks_match_num=75, float Lowe_ratio = 0.75,
                    int out_Max = 20000, str FeauterType = 'ORB', int UpperLimitOfFeats = 30000):

        self.watch = watch()
        if len(direct)==0:
            raise IOError("Please Enter The Valid Images Directory")
        else:self.direct = direct

        self.all_poses,self.all_vec_pose,self.Rs,self.Ts,self.Instrinc_Mat,self.Destoration = GetMatrices(self.direct)

        image_names = glob.glob(direct+"/*.png")
        self.image_names = sorted(image_names,key=self.ArrangeNames)
        self.SiftStep = SiftStep
        self.FeauterType = FeauterType
        self.UpperLimitOfFeats = UpperLimitOfFeats
        if FeauterType == 'SIFT':
            self.activeFeat = cv2.xfeatures2d.SIFT_create(out_Max)
            # FLANN parameters
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=checks_match_num) # or pass empty dictionary
            self.activeFlann = cv2.FlannBasedMatcher(index_params,search_params)

        elif FeauterType == 'ORB':
            self.activeFeat = cv2.ORB_create(out_Max)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 12,
                                key_size = 20,
                                multi_probe_level = 2,
                                trees = 3)

            search_params = dict(checks=checks_match_num) # or pass empty dictionary
            self.activeFlann = cv2.FlannBasedMatcher(index_params,search_params)


        self.lowe_ratio = Lowe_ratio

        #np.save('allP',np.array(self.all_poses))
        self.pointsCloud   = []
        #self.nodesAndProjs = []
        self.Image_Index = 0
        self.PointsFile = file('pointsNew.txt','w+')
        self.ProjsFile = file('NodesProjs.txt','w+')
        self.OptFlow = MatchAndAddCorners()
        self.PointsBankActive = []
        self.PointsBankNonActive = []

        self.Rects = [[],[]]
        #self.pool2go = Pool(3)

        self.CurrPrjErr = []

    def ArrangeNames(self,x):
        return int(x[len(self.direct)+1:-4])

    cpdef DetectDenseORB(self,cnp.ndarray[UINT8, ndim=3] img,unsigned int step_size,list rect):
        #kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
        #                                for x in range(0, gray.shape[1], step_size)]
        cdef list kps_in, kps_out = []
        cdef cnp.ndarray[UINT8, ndim=2] mask, Ds_np
        cdef unsigned int N
        if step_size:
            grid   = GiveGrid(step_size,rect) 
            kps_in = map(MakeKpsGrid,grid)

            mask = np.ones((img.shape[0], img.shape[1]),dtype=np.uint8) 
            mask[rect[1]:rect[3],rect[0]:rect[2]] = 0
        else:
            kps_in = []
            kps_out = self.activeFeat.detect(img,None)
        if not(kps_out): kps_out = self.activeFeat.detect(img,mask)

        N = len(kps_in)+len(kps_out)
        print('Number of Kps {}'.format(N))
        """
        if N>self.UpperLimitOfFeats:
            Ks,Ds = self.activeFeat.compute(img,kps_in+kps_out)
            #take the most variance
            #with nogil:
            sortedKsDs = sorted(zip(Ks,Ds),key=Sort2MostVar)
            Ks_,Ds_ = zip(*sortedKsDs[-self.UpperLimitOfFeats:])
            
            Ks_,Ds_ = [], []
            for KD in sortedKsDs[-self.UpperLimitOfFeats:]:
                Ks_.append(KD[0])
                Ds_.append(KD[1])
            
            # for ORB here
            Ds_np = np.asarray(Ds_,dtype=np.uint8)
            Ks_list = list(Ks_)
            return Ks_list , Ds_np
        """
        return kps_in+kps_out

    cpdef DetectComputeDenseSift(self,cnp.ndarray[UINT8, ndim=3] img,unsigned int step_size,list rect):
        #kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
        #                                for x in range(0, gray.shape[1], step_size)]
        cdef list kps_in,Ks,sortedKsDs,Ks_list, kps_out = []
        cdef cnp.ndarray[UINT8, ndim=2] mask, Ds_np, Ds
        cdef unsigned int N
        if step_size:
            grid   = GiveGrid(step_size,rect) 
            kps_in = map(MakeKpsGrid,grid)

            mask = np.ones((img.shape[0], img.shape[1]),dtype=np.uint8) 
            mask[rect[1]:rect[3],rect[0]:rect[2]] = 0
        else:
            kps_in = []
            kps_out = self.activeFeat.detect(img,None)
        if not(kps_out): kps_out = self.activeFeat.detect(img,mask)

        N = len(kps_in)+len(kps_out)
        print('Number of Kps {}'.format(N))

        if N>self.UpperLimitOfFeats:
            Ks,Ds = self.activeFeat.compute(img,kps_in+kps_out)
            #take the most variance
            #with nogil:
            sortedKsDs = sorted(zip(Ks,Ds),key=Sort2MostVar)
            Ks_,Ds_ = zip(*sortedKsDs[-self.UpperLimitOfFeats:])
            """"
            Ks_,Ds_ = [], []
            for KD in sortedKsDs[-self.UpperLimitOfFeats:]:
                Ks_.append(KD[0])
                Ds_.append(KD[1])
            """
            # for ORB here
            Ds_np = np.asarray(Ds_,dtype=np.uint8)
            Ks_list = list(Ks_)
            return Ks_list , Ds_np
        return self.activeFeat.compute(img,kps_in+kps_out)
        

    def DetectComputeHDenseSift(self,gray,step_size):

        kp  = self.activeFeat.detect(gray)
        kp += [cv2.KeyPoint(x, y, step_size) for y in xrange(0, gray.shape[0], step_size)
                                        for x in xrange(0, gray.shape[1], step_size)]
        #kp = self.sift.detect(gray)
        print(len(kp),' Number of Kps')
        return self.activeFeat.compute(gray,kp)

    cpdef list FindRect(self,cnp.ndarray[UINT8, ndim=3] img1,list rect,cnp.ndarray[UINT8, ndim=3] img2):

        cdef cnp.ndarray[UINT8, ndim=2] I1,I2,temp 
        cdef cnp.ndarray[FLOAT, ndim=2] res
        cdef unsigned int w,h
        cdef float min_val, max_val
        cdef (int,int) bottom_right, min_loc, max_loc
        I1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        I2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

        # rect is x,y,x*,y* putten y:y*,x:x*
        # rect = map(int,rect1)

        temp = I1[rect[1]:rect[3],rect[0]:rect[2]]
        h,w = temp.shape[0],temp.shape[1]

        # methode is cv2.TM_CCOEFF (4) or Normalized (5)
        res = cv2.matchTemplate(I2,temp,5)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print('max val is:' + str(max_val))
        if max_val < 0.65:
            k = SelectROI(self.image_names[self.Image_Index+1])
            Rects = map(int,k.rect)
            return Rects
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # same as rect x,y,x1,y1
        return [top_left[0] , top_left[1] , bottom_right[0] , bottom_right[1]]

    def FirstTwo(self,Draw_matches):
        I1 = cv2.imread(self.image_names[0],1)
        I2 = cv2.imread(self.image_names[1],1)
        """
        I1 = cv2.undistort(I1,
            #cv2.imread(self.image_names[0],0),
                           self.Instrinc_Mat,self.Destoration)

        I2 = cv2.undistort(I2,
            #cv2.imread(self.image_names[1],0),
                           self.Instrinc_Mat,self.Destoration)
        """
        #for Drawing porpouses:
        self.I_last = I2.copy()

        if self.SiftStep: self.Rects[1] = self.FindRect(I1,self.Rects[0],I2)
#####################################

        kp1  = self.DetectDenseORB(I1,self.SiftStep,self.Rects[0])
        #kp2, des2 = self.DetectComputeDenseSift(I2,self.SiftStep,self.Rects[1])
        # ps is in float 32 in N*2
        #MatchesIdx is N*2 query train
        p1,p2,MatchesIdx = self.OptFlow.Match(I1,I2,kp1)

        i = 0
        firstImagePoints  = []
        secondImagePoints = []
        good              = []
        kp1               = []
        kp2_d             = []
        Radius            = 5
        if self.SiftStep: Radius = self.SiftStep
        for p_1,p_2 in zip(p1,p2):
            kp_1   = cv2.KeyPoint(p_1[0],p_1[1],Radius)
            kp_2_d = cv2.KeyPoint(p_2[0],p_2[1],Radius)
            kp1.append(kp_1)
            kp2_d.append(kp_2_d)
            firstImagePoints.append(pointStruct(kp_1,np.array([0],np.uint8),0))
            secondImagePoints.append(pointStruct(kp_2_d,np.array([0],np.uint8),1))
            good.append(cv2.DMatch(i,i,1,1))#can we add match errors for distance??
            i = i + 1

        kp2_added = self.DetectMorePoints(I2,p2)
        for kp_2 in kp2_added:
            secondImagePoints.append(pointStruct(kp_2,np.array([0],np.uint8),1))
        kp2 = kp2_d + kp2_added
        #good = [m for m,n in matches if m.distance < self.lowe_ratio *n.distance]
        #src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        #dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        #_, mask = cv2.findFundamentalMat(src_pts, dst_pts,cv2.FM_RANSAC ,
        #                                 Max_Error_Allowed,0.99)

        #good_after_ransac=[[good[i]] for i,x in enumerate(mask) if x]


######################################
        #bp()
        if Draw_matches:
            I12=cv2.drawMatches(I1,kp1,I2,kp2,good,np.vstack((I1,I2)))
            cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
            cv2.imshow('pic',I12)
            cv2.waitKey(0)

        ptsrc = cv2.triangulatePoints(self.all_poses[0],self.all_poses[1],p1.T,p2.T)

        points_cloud = cv2.convertPointsFromHomogeneous(ptsrc.T)
        points_cloud  = np.transpose(strip_out(np.transpose(points_cloud)))
        #points_cloud = np.array(self.pool2go.map(lambda x: x[0],points_cloud.T)).T
        #points_cloud = np.array([z[0] for z in points_cloud.T]).T
        good3DPointsIndx = self.ReprojectionError(points_cloud,p1.T,p2.T)

        good_2 = get_good_2(good,good3DPointsIndx)
        self.CurrPrjErr = get_good_2(self.CurrPrjErr,good3DPointsIndx)
        #[p for i,p in enumerate(good) if good3DPointsIndx[i]]
        
        self.PointsBankActive = secondImagePoints
        for c,m in enumerate(good_2):
            pntS = firstImagePoints[m.queryIdx]
            mpntS= self.PointsBankActive[m.trainIdx]
            kp,des,i = mpntS.returnKDActive()
            pntS.addProj(kp,des,i, self.CurrPrjErr[c])
            self.PointsBankActive[m.trainIdx] = pntS
            
        self.pointsFilter()
        
        print('Number of aduqate Points {}'.format(len(good_2)))
        #good_after_ransac = good_after_ransac[good3DPointsIndx]

        self.Image_Index += 1

    cpdef bint AddImage(self,cnp.ndarray[UINT8, ndim=3] I,bint Draw_matches):

        #I = cv2.undistort(I,self.Instrinc_Mat,self.Destoration)
        cdef list kpn, kp1,des0, matches, good, addedImagePoints, good3DPointsIndx
        cdef list good_2, addedImagePointsCopy, tempPntsMat
        cdef cnp.ndarray[cnp.npy_bool, ndim=1, cast=True]  addedImagePointsIdx,PointsActiveIdx
        cdef bint y
        cdef cnp.ndarray[FLOAT, ndim=2] p1,p2
        cdef cnp.ndarray[FLOAT, ndim=3] points_cloud3
        cdef cnp.ndarray[FLOAT, ndim=2] points_cloud
        cdef unsigned int i,t,q,c
        cdef cnp.ndarray[UINT8, ndim=3, cast=True] I12
        #for ORB
        #cdef cnp.uint8_t[:,:] des1
        cdef cnp.ndarray[UINT8, ndim=2] desn , des1
        cdef cnp.ndarray[UINT8, ndim=1] des
        cdef cnp.ndarray[FLOAT, ndim=2] ptsrc

        if self.SiftStep: self.Rects.append(self.FindRect(np.asarray(self.I_last,dtype=np.uint8),self.Rects[-1],I))
####################################
        #kpn, desn = self.DetectComputeDenseSift(I,self.SiftStep,self.Rects[-1])

        kp1     = []
        for k in self.PointsBankActive:
            kp,_,_ = k.returnKDActive()
            kp1.append(kp)
        p1,p2,MatchesIdx = self.OptFlow.Match(np.asarray(self.I_last,dtype=np.uint8),I,kp1)

        i       = 0
        good    = []
        kpn     = []
        addedImagePoints = []
        Radius            = 5
        if self.SiftStep: Radius = self.SiftStep
        for MIdx,p_2 in zip(MatchesIdx,p2):
            kp_2_d = cv2.KeyPoint(p_2[0],p_2[1],Radius)
            kpn.append(kp_2_d)
            addedImagePoints.append(pointStruct(kp_2_d,np.array([0],np.uint8),self.Image_Index+1))
            good.append(cv2.DMatch(MIdx[0],MIdx[1],1,1))#can we add match errors for distance??(q,t)

        kp2_added = self.DetectMorePoints(I,p2)
        for kp_2 in kp2_added:
            addedImagePoints.append(pointStruct(kp_2,np.array([0],np.uint8),self.Image_Index+1))
        kpn = kpn + kp2_added



        print('**************0')

        print('**************1')

        #with nogil:
        #matches = self.activeFlann.knnMatch(des1,desn,2)
        print('**************2')

        # adding dense sift if match es aren't enough
        """
        if len(good) < 50:
            kpn, desn = self.DetectComputeHDenseSift(I,self.SiftStep+2)
            matches = self.flann.knnMatch(des1,desn,k=2)
            good = []
            for mn in matches:
                if len(mn)==1:good += mn
                elif len(mn)!=2: continue
                elif mn[0].distance < 0.75*mn[1].distance: good += [mn[0]]
        """
        #if(self.Image_Index>=80):bp()
        print('**************3')
        addedImagePointsIdx = np.ones(len(addedImagePoints),dtype=np.bool)
        PointsActiveIdx     = np.ones(len(self.PointsBankActive),dtype=np.bool)

        #good = [m for m,n in matches if m.distance < 0.75*n.distance]

        #src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        #dst_pts = np.float32([ kpn[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        #_, mask = cv2.findFundamentalMat(src_pts, dst_pts,cv2.FM_RANSAC ,
        #                                 Max_Error_Allowed,0.99)
        #good_after_ransac=[[good[i]] for i,x in enumerate(mask) if x]

        #good_after_ransac=[[i] for i in good]
        print('**************4')
        p1 = p1.T#np.float32([ kp1[m.queryIdx].pt for m in good ]).T
        p2 = p2.T#np.float32([ kpn[m.trainIdx].pt for m in good ]).T
#############################################

        ptsrc = cv2.triangulatePoints(self.all_poses[self.Image_Index],
                                      self.all_poses[self.Image_Index+1],p1,p2)
        
        points_cloud3 = cv2.convertPointsFromHomogeneous(ptsrc.T)
        points_cloud  = np.transpose(strip_out(np.transpose(points_cloud3)))
        #points_cloud = np.array([z[0] for z in points_cloud.T]).T
        good3DPointsIndx = self.ReprojectionError(points_cloud,p1,p2)
        print('**************5')
        #good_2 = [p for i,p in zip(good3DPointsIndx,good) if i]
        good_2 = get_good_2(good,good3DPointsIndx)
        self.CurrPrjErr = get_good_2(self.CurrPrjErr,good3DPointsIndx)
        #good_after_ransac = good_after_ransac[good3DPointsIndx]
        #newPointsBankActive = []
        for c,m in enumerate(good_2):
            q,t = m.queryIdx,m.trainIdx
            originalP = self.PointsBankActive[q]
            otherP = addedImagePoints[t]
            k,d,i = otherP.returnKDActive()
            originalP.addProj(k,d,i,self.CurrPrjErr[c])
            #newPointsBankActive.append(originalP)
            #self.PointsBankActive[m.queryIdx] = originalP
            addedImagePointsIdx[t] = False
            PointsActiveIdx[q]     = False
        print('**************6')
        tempPntsMat = []
        for x,y in zip(self.PointsBankActive,PointsActiveIdx):
            if y:
                self.PointsBankNonActive.append(x)
            else: 
                tempPntsMat.append(x)
        #self.PointsBankNonActive.extend([x for x,y in zip(self.PointsBankActive,PointsActiveIdx) if y])
        #tempPntsMat = [x for x,y in zip(self.PointsBankActive,PointsActiveIdx) if not(y)]
        self.PointsBankActive = tempPntsMat
        addedImagePointsCopy = [x for x,y in zip(addedImagePoints,addedImagePointsIdx) if y]
        self.PointsBankActive.extend(addedImagePointsCopy)
        #self.pointsFilter()

        print('Number of aduqate Points {}'.format(len(good_2)))

        #bp()#self.TestPointsVsF(p1,p2)
        cv2.destroyAllWindows()
        if Draw_matches:
            I12=cv2.drawMatches(np.asarray(self.I_last,dtype=np.uint8),kp1,I,kpn,good_2,np.vstack((self.I_last,I)))
            cv2.namedWindow('pic'+str(len(good_2)), cv2.WINDOW_NORMAL)
            cv2.imshow('pic'+str(len(good_2)),I12)
            k=cv2.waitKey(300)
        if k==27 & 0xFF:return True
        print('**************7')
        self.I_last = I.copy()
        self.Image_Index += 1

        return False

    def TringulateAll(self,ReprojectThresh,Draw_matches=False,load_auto ="",
                    tracks_diffs = 1, min_projections = 3, FinalErrPrj = 1.0):

        cdef cnp.ndarray[FLOAT, ndim=1] Prslt
        if self.SiftStep:
            k = SelectROI(self.image_names[0])
            self.Rects[0] = map(int,k.rect)
        if load_auto:
            self.PointsBankNonActive = np.load(load_auto)
            for x in xrange(len(self.PointsBankNonActive)):
                self.PointsBankNonActive[x].prop2retrive()
        else:
            self.ReprojectThresh = ReprojectThresh
            self.FirstTwo(Draw_matches)
            for img in self.image_names[2:]:

                print("adding image"+img)
                #I = cv2.cvtColor(cv2.imread(img,1),cv2.COLOR_BGR2YUV)
                I = cv2.imread(img,1)
                if self.AddImage(I,Draw_matches):
                    break
            cv2.destroyAllWindows()
            print("Tracking Points is over , Now to 3D")
            # some clean up
            for ps in self.PointsBankActive:
                if ps.numberOfProjs > 1:
                    self.PointsBankNonActive.append(ps)
            Points2save = np.array(self.PointsBankNonActive[:]).copy()
            #bp()
            """
            for x in range(len(self.PointsBankNonActive)):
                Points2save[x].prop2save()
                #self.PointsBankNonActive[x].prop2retrive()
            np.save("LastResult",Points2save)
            for x in range(len(self.PointsBankNonActive)):
                Points2save[x].prop2retrive()
            """
        self.nodesAndProjs = [[] for x in self.image_names]
        m = 1
        #N_true = 0
        for ps in self.PointsBankNonActive:
            if (ps.numberOfProjs < min_projections):
                if len(ps.Errors)==0:continue
                elif (np.mean(ps.Errors)>FinalErrPrj):continue
                # this is bug, pnts from second image!
                #print("Really still zeros here with {} projection".format(ps.numberOfProjs))
                
            Prslt = ps.track23D(self.all_poses,tracks_diffs)
            if any(Prslt):
                self.pointsCloud.append(Prslt)
            #print(cv2.triangulatePoints(self.all_poses[0],self.all_poses[1],np.float32([ps.kpStruct[0].pt]).T,np.float32([ps.kpStruct[1].pt]).T))
            #print(ps.track2d3(self.Ts,self.Rs,self.Instrinc_Mat,2))
            #self.pointsCloud.append(ps.track2d3(self.Ts,self.Rs,self.Instrinc_Mat,1))
            #add projections form ps

                for j,i in enumerate(ps.imagesIndices):
                    self.nodesAndProjs[i].append([m]+ps.kpPositions[j])
                m += 1
        N = len(self.pointsCloud)
        print("Tringluate Complete with {0} Points from {1} Cameras".format(N,
              self.Image_Index+1))

    def WriteNodes3DPoints(self):
        #all zero as start
        self.PointsFile.writelines(str(np.zeros((1,12))[0])[1:-1])
        for p in self.pointsCloud:
            #No covariance
            p = list(p)
            p.extend([0,0,0,0,0,0,0,0,0])
            self.PointsFile.writelines('\n')
            self.PointsFile.writelines(" ".join(map(str,p)))
        self.PointsFile.close()
        #Note: Last line here is \n i.e: empty
        for i,p in enumerate(self.nodesAndProjs):
            self.ProjsFile.writelines('P'+str(i+1)+'\n')
            self.ProjsFile.writelines("\n".join(map(str,self.all_vec_pose[i])))
            self.ProjsFile.writelines('\n')
            for pInImg in p:
                self.ProjsFile.writelines(" ".join(map(str,pInImg)))
                self.ProjsFile.writelines('\n')
        self.ProjsFile.close()
        xyz = np.float32(self.pointsCloud)
        #pcloud = pcl.PointCloud_PointXYZRGB()
        pcloud_nocolor = pcl.PointCloud()
        Magnifize = 10
        xyz*=Magnifize
        #mean_c = np.float32([[(int(c)<<16)|(int(c)<<8)|int(c)] for c in self.D3_points_index[2][0]])
        #bp()
        #xyzc = np.hstack((xyz,mean_c))

        #pcloud.from_array(xyzc)
        pcloud_nocolor.from_array(xyz)
        #pcloud.to_file(self.direct+'/xyz_cloud.pcd')
        pcloud_nocolor.to_file(self.direct+'/xyz_cloud_nocolor.xyz')
        print("Time taken is : {0} second".format(watch() - self.watch))

    cpdef list ReprojectionError(self,cnp.ndarray[FLOAT, ndim=2] Point_cloud,
                cnp.ndarray[FLOAT, ndim=2] pnt1, cnp.ndarray[FLOAT, ndim=2] pnt2):
        cdef unsigned int  n,m,npts
        cdef float NewThresh,x
        cdef cnp.ndarray[FLOAT, ndim=2] pn1,pn2 
        cdef cnp.ndarray[FLOAT, ndim=3] pnr1,pnr2
        cdef list errors, Result
        n = self.Image_Index
        m = self.Image_Index + 1
        #Todo : do u rellay need Destoration if aleardy undistrored
        pnr1,_ = cv2.projectPoints(Point_cloud,self.Rs[n],self.Ts[n],
                                   np.array(self.Instrinc_Mat),np.array([[0.0]*5]))
        pnr2,_ = cv2.projectPoints(Point_cloud,self.Rs[m],self.Ts[m],
                                   np.array(self.Instrinc_Mat),np.array([[0.0]*5]))

        pn1 = np.transpose(strip_out(pnr1))
        pn2 = np.transpose(strip_out(pnr2))

        errors = calc_errors(pn1,pnt1,pn2,pnt2)
        """
        o = (pn1- pnt1)**2
        a = np.sqrt(map(lambda x: x[0]+x[1],o.T))

        o = (pn2- pnt2)**2
        b = np.sqrt(map(lambda x: x[0]+x[1],o.T))

        errors = np.mean(np.vstack((a,b)),axis=0)
        """
        self.CurrPrjErr = errors
        Result = [x<=self.ReprojectThresh for x in errors]
        #Result = errors<=self.ReprojectThresh
        if sum(Result)<20:
            #no less than 40pts
            npts = [40,len(errors)-1][len(errors)<41]
            NewThresh = sorted(errors)[npts]
            print("New Thresh is: "+str(NewThresh))
            #return errors <NewThresh
            return [x<=NewThresh for x in errors]
        else:
            return Result

#New_point_clouds = Point_cloud[errors<thresh]
####################################################
### All the following methods for Future Development:
####################################################
        
    def pointsFilter(self):
        newPointsBankActive = []
        for ps in self.PointsBankActive:
            if ps.imagesIndices[-1] == self.Image_Index+1 :
                newPointsBankActive.append(ps)
                continue
            elif ps.numberOfProjs > 1:
                self.PointsBankNonActive.append(ps)
        self.PointsBankActive = newPointsBankActive

    def DetectMorePoints(self,I,Pkp_p):
        #without mask
        N_2_More = 2*(self.UpperLimitOfFeats - len(Pkp_p))
        Pkp_p.reshape((Pkp_p.shape[0],1,Pkp_p.shape[-1])) 
        Feats = cv2.ORB_create(N_2_More)

        if self.SiftStep:
            grid   = GiveGrid(self.SiftStep,self.Rects[-1]) 
            kps_out = map(MakeKpsGrid,grid)
            Radius = self.SiftStep
        else:
            kps_out = Feats.detect(I,None)
            Radius = 6#[6,self.SiftStep][bool(self.SiftStep)]
        Ks_result = []
        i,c,f = 0,0,len(kps_out)
        while (i < self.UpperLimitOfFeats) and (c < f):
            k = kps_out[c]
            c = c+1
            if np.any(sp.distance.cdist(np.array([k.pt]),Pkp_p) < Radius):
                #pt not okay
                continue
            else:
                Ks_result.append(k)
                i = i+1
        return Ks_result
