import numpy as np

class ReadPoses():

    def __init__(self, dirc):
        self.direct = dirc
        self.Ps = []
        if self.ReadInstrinc():
            self.Ps = self.GetPoses()

    def ReadInstrinc(self):
        self.Instrinc_Mat = np.array([[2217.018764764748,0.0,1280.5],[0,2217.01876476474,960.5],[0,0,1]])
        self.Destoration = np.array([0,0,0,0,0])
        return True

    def GetPoses(self):
        all_poses,take7,pose,self.all_vec_pose=[],0,[],[]
        self.Rs,self.Ts = [],[]
        with open(self.direct+"/nodes_and_prjcts2.txt") as f:
            for line in f:
                if take7:
                    pose +=[float(line)]
                    take7-=1
                if 'P' in line:
                    take7=7
                    if pose:
                        all_poses += [self.GetPFromPose(pose)]
                        self.all_vec_pose+=[np.array(pose)]
                    pose=[]
        if pose:
            all_poses+=[self.GetPFromPose(pose)]
            self.all_vec_pose+=[np.array(pose)]
        return all_poses

    def GetPFromPose(self,pose_vec):
        #it was: w,x,y,z
        w,x,y,z =pose_vec[3:]
        qnorm = np.sqrt(sum(map(lambda x: x**2,[w,x,y,z])))
        w,x,y,z = w/qnorm, x/qnorm, y/qnorm, z/qnorm
        R = np.array([[1-2*(y**2+z**2) , 2*(x*y-z*w)   ,  2*(x*z+y*w)  ],
                      [ 2*(x*y+z*w)    ,1-2*(x**2+z**2),  2*(y*z-x*w)  ],
                      [ 2*(x*z-y*w)    , 2*(y*z+x*w)   ,1-2*(x**2+y**2)]]).T
        self.Rs.append(R)

        T = np.array([[pose_vec[0]],[pose_vec[1]],[pose_vec[2]]])
        T = -R.dot(T)
        self.Ts.append(T)
        P = np.hstack((R,T))
        return self.Instrinc_Mat.dot(P)

    def cross4(self,a,b):
    	"""
    	cross product for quatrion
    	"""
    	r1,x1,y1,z1 = a

    	res=np.array([[r1, -x1, -y1, -z1],
    		    [x1,  r1, -z1,  y1],
    		    [y1,  z1,  r1, -x1],
    		    [z1, -y1,  x1,  r1]])
    	return res.dot(np.array(b))

    def quat_from_odo_2_cam(self,qxyzw):
        qwxyz = np.hstack((qxyzw[-1],qxyzw[:3]))
        sq2 = 1/np.sqrt(2)
        q90y  = np.array([sq2,0,sq2,0])
        qn90z = np.array([sq2,0,0,-sq2])

        qn90x = np.array([sq2,sq2,0.0,0.0])
        q180z = np.array([0.0,0.0,0.0,1.0])

        qall  = np.array([0.0,0.000, -sq2, sq2])
        #return self.cross4(self.cross4(qwxyz,q90y),qn90z)
        #return self.cross4(self.cross4(qwxyz,qn90x),q180z)
        return self.cross4(qwxyz,qall)

    def ReturnMatrices(self):
        return self.Ps,self.all_vec_pose,self.Rs,self.Ts,self.Instrinc_Mat,self.Destoration

    def write_poses_to_file(self):
        pass
        i = 1
        some_file = open("nodes_and_prjcts.txt",'w')
        for p in self.all_vec_pose:
            some_file.writelines("P"+str(i))
            some_file.writelines("\n")
            some_file.writelines('\n'.join(map(str,p)))
            some_file.writelines("\n")
            #falsy points
            some_file.writelines("0 0 0\n")
            some_file.writelines("\n")
            i += 1
        some_file.close()




def GetMatrices(TheDir):
    c = ReadPoses(TheDir)
    return c.ReturnMatrices()
