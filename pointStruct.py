import numpy as np
import cv2
from pdb import set_trace as bp

class pointStruct():
    
    def __init__(self,kp,des,i):

        self.kpPositions   = [[int(kp.pt[0]),int(kp.pt[1])]]
        self.kpStruct      = [kp]
        self.descriptors   = [des]
        self.imagesIndices = [i]
        self.numberOfProjs =  1
        self.cposes        = []
        self.angels        = []
    
    def addProj(self,kp,des,i):
        self.kpPositions.append([int(kp.pt[0]),int(kp.pt[1])])
        self.kpStruct.append(kp)
        self.descriptors.append(des)
        self.imagesIndices.append(i)
        self.numberOfProjs += 1

    def track2d3(self,tvecs,rmats,K,covThresh):
        for i,kp in enumerate(self.kpStruct):
            imgI = self.imagesIndices[i]
            self.addMeasurment(tvecs[imgI],rmats[imgI],kp.pt,K)
        pose,cov = self.computeState()
        a,b=cv2.minMaxLoc(cov)[:2]
        if max(abs(a),abs(b)) < covThresh:
            print("cov is: "+str(max(abs(a),abs(b))))
            return pose.T[0].tolist()
        else:
            print("too much noise, deleteing point..")
            return np.array([0,0,0])
        

    def track23D(self,Poses, tracks_max_diff = 1):
        """
        return 3 numbers: x,y,z
        according to some alogarithem
        TODO: better methode
        TODO: add Covraince
        """
        all_points = np.array([0,0,0])
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
                p3d = cv2.triangulatePoints(Poses[i],Poses[j],
                                      np.float32([self.kpStruct[m].pt]).T,
                                      np.float32([self.kpStruct[n+m+1].pt]).T)
                p3d = cv2.convertPointsFromHomogeneous(p3d.T)
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
        if Err>tracks_max_diff : return  np.array([0,0,0])#all_points[2]
        #
        print("Err is: " + str(Err))
        return np.mean(all_points[1:],axis=0)#all_points[np.argmax(spaces)+1]## self.TringulateNView(Xs,Ps)#
    
    def possibleColor(self):
        pass
    
    def returnKpDes(self):
        #as lists
        return self.kpStruct,self.descriptors
    
    def returnKDActive(self):
        return self.kpStruct[-1],self.descriptors[-1],self.imagesIndices[-1]
    
    def addMeasurment(self,tvec,rvec,pt,K):
        los = np.array([[pt[0]],[pt[1]],[1]])
        self.cposes.append(-rvec.T.dot(tvec))
        los = rvec.T.dot(los)
        azel = self.getAzEl(np.zeros((3,1)),los)
        self.angels.append(azel)

    def getAzEl(self,pstart,pend):
        diff = pend-pstart
        x = np.arctan2(diff[1],diff[0])
        y = np.arcsin(diff[2]/cv2.norm(diff))
        return [x,y]

    def computeState(self):
        N = self.numberOfProjs
        F = np.zeros((2*N,3))
        b = np.zeros((2*N,1))
        for i in range(N):
            F[ i ] = np.array([np.sin(self.angels[i][0])[0], -np.cos(self.angels[i][0])[0] , 0])
            F[N+i] = np.array([-np.sin(self.angels[i][1])*np.cos(self.angels[i][0]),
                               -np.sin(self.angels[i][1])*np.sin(self.angels[i][0]),
                                np.cos(self.angels[i][1])]).T

            b[ i ] = F[ i ].dot(self.cposes[i])[0]
            b[N+i] = F[N+i].dot(self.cposes[i])[0]
        iple = np.array([])
        _,iple=cv2.solve(F,b,iple,cv2.DECOMP_SVD)
        G = np.zeros((2*N,3))
        W = np.zeros((2*N,2*N))

        for i in range(N):
            new_ang = self.getAzEl(self.cposes[i],iple)
            G[ i ] = np.array([ np.sin(new_ang[0]),-np.cos(new_ang[0]) , 0]).T
            G[N+i] = np.array([-np.sin(new_ang[1])*np.cos(new_ang[0]),
                               -np.sin(new_ang[1])*np.sin(new_ang[0]),
                                np.cos(new_ang[1])]).T
            diff3 = iple - self.cposes[i]
            diff2 = diff3[:2]
            W[ i ][ i ] = cv2.norm(diff2)
            W[N+i][N+i] = cv2.norm(diff3) - W[i][i]
        A = np.linalg.inv(G.T.dot(W).dot(F))
        B = G.T.dot(W).dot(b)
        state = A.dot(B)

        cov = (G.T.dot(F))/(N/2.0)
        cov *= np.sum(F.dot(state)-b)
        return state,cov

    def prop2save(self):
        r = []
        for kp in self.kpStruct:
            r.append(kp.pt)
        self.kpStruct = r
    def prop2retrive(self):
        r = []
        for kp in self.kpStruct:
            r.append(cv2.KeyPoint(kp[0],kp[1],1))
        self.kpStruct = r


    def TringulateNView(self,Xs,Ps):
        """
        Xs points (numpy array) each one in a view:
        [[x1 x2 x3 .. ]]
        [[y1 y2 y3 ..]]
        Ps is a list of P(3*4) numpy array

        output: 3*1 of 3d point
        """

        if Xs.shape[0]!=2 or len(Ps)!=Xs.shape[1]:
            print("Please insert 2*N matrix Point and 3*4 list")
            return False
        n = Xs.shape[1]
        design = np.zeros((3*n,4+n))
        for i in range(n):
            for jj in range(3):
                for ii in range(4):
                    design[3*i+jj][ii] = -Ps[i][jj][ii]
            design[3*i + 0][4 + i] = Xs[0][i]
            design[3*i + 1][4 + i] = Xs[1][i]
            design[3*i + 2][4 + i] = 1.0

            #out_raw = cv2.solve(design)
            e_vals, e_vecs = np.linalg.eig(np.dot(design.T,design))#
            out = e_vecs[:,np.argmin(e_vals)][:4]
            #out = out_raw[:4]
            out_f = out/out[-1]
        return out_f[:-1]
