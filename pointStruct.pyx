import numpy as np
import cv2
cimport numpy as cnp


ctypedef unsigned char SMALL_INT
ctypedef cnp.float32_t FLOAT
ctypedef cnp.uint8_t UINT8

cimport cython

@cython.boundscheck(False)
cdef class pointStruct():
    cdef public list kpPositions,kpStruct,descriptors,imagesIndices, Errors
    cdef public unsigned int numberOfProjs
    # for ORB
    def  __init__(self,kp,cnp.ndarray[UINT8, ndim=1] des,int i):
        self.kpPositions   = [[int(kp.pt[0]),int(kp.pt[1])]]
        self.kpStruct      = [kp]
        self.descriptors   = [des]
        self.imagesIndices = [i]
        self.numberOfProjs =  1
        self.Errors        = []

    #def init_kpoint(self,):
    
    cpdef addProj(self,kp, cnp.ndarray[UINT8, ndim=1]  des,int i ,float Proj_Err):
        self.kpPositions.append([int(kp.pt[0]),int(kp.pt[1])])
        self.kpStruct.append(kp)
        self.descriptors.append(des)
        self.imagesIndices.append(i)
        self.numberOfProjs += 1
        self.Errors.append(Proj_Err)

    cpdef cnp.ndarray[FLOAT, ndim=1] track23D(self,list Poses,double tracks_max_diff = 1):
        """
        return 3 numbers: x,y,z
        according to some alogarithem
        TODO: better methode
        TODO: add Covraince
        """
        cdef cnp.ndarray[FLOAT, ndim=2] all_points,p2d
        cdef cnp.ndarray[FLOAT, ndim=3] p3d
        cdef list spaces
        cdef unsigned int m,i,n,j
        cdef FLOAT Err
        all_points = np.array([[0,0,0]],dtype=np.float32)
        spaces = []
        #first = True
        for m,i in enumerate(self.imagesIndices):
            for n,j in enumerate(self.imagesIndices[m+1:]):
                """
                if first:
                    Ps = [Poses[i]]
                    Xs = np.float32([self.kpStruct[m].pt]).T
                    first = False
                Ps.append(Poses[j])
                Xs = np.hstack((Xs,np.float32([self.kpStruct[n+m+1].pt]).T))
                """
                p2d = cv2.triangulatePoints(Poses[i],Poses[j],
                                      np.float32([self.kpStruct[m].pt]).T,
                                      np.float32([self.kpStruct[n+m+1].pt]).T)
                p3d = cv2.convertPointsFromHomogeneous(p2d.T)
                if np.any((abs(p3d)>0.00001)):
                    #p3d[0][0] is 1,3 shape
                    #spaces.append(abs(i-j))
                    all_points = np.vstack((all_points,p3d[0][0]))

        if len(all_points[1:])-1:
            #more than 1 pt
            Err = np.max(np.sum(abs(np.diff(all_points[1:],axis=0)),axis=1))
        else:
            # 1 pt, .. care
            return  all_points[1] # np.array([0,0,0])
        # care: and (max(spaces)<8)
        if Err>tracks_max_diff : return  np.array([0,0,0],dtype=np.float32)#all_points[2]
        #
        print("Err is: " + str(Err))
        return np.mean(all_points[1:],axis=0)#all_points[np.argmax(spaces)+1]## self.TringulateNView(Xs,Ps)#
    
    
    cpdef  returnKpDes(self):
        #as lists
        return self.kpStruct,self.descriptors
    
    cpdef  returnKDActive(self):
        return self.kpStruct[-1],self.descriptors[-1],self.imagesIndices[-1]
    
    def prop2save(self):
        cdef list r = []
        for kp in self.kpStruct:
            r.append(kp.pt)
        self.kpStruct = r
    def prop2retrive(self):
        cdef list r = []
        for kp in self.kpStruct:
            r.append(cv2.KeyPoint(kp[0],kp[1],1))
        self.kpStruct = r