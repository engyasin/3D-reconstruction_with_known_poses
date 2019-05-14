# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:45:14 2018

@author: Eng-Yasin
"""

from Tringulate__Nonnumba import ImagesWithPoses

#def main():

# Always with ORB or change to float32

TheDir = "./Turtle_images_sim"

#                    Sift_Steps(or ORB /600) if 0 means no ROI,  
c = ImagesWithPoses(        7               ,
                    #    Images_Dir      , Flann_matching_param,
                           TheDir        ,         100         ,
                    #       Lowe Ratio  , max number of pts outside ,  FeauterType 
                           0.75          ,          20000            ,    'ORB'    )

#                ReprojectThresh,       Draw_flag   , eggs        
c.TringulateAll(        15      ,       True        ,  ""         ,
                #,max diffs between >2 projections ,  take points only with this many projections
                           0.45                     ,                      3                     )


#c.DrawCloud()
c.WriteNodes3DPoints()

#if __name__=="__main__":
#    main()

