# 3D-Reconstruction with known poses

This project is solving the problem of 3D reconstruction from 2D images with the following restrictions:

* the calibration parameters are known.

* the extrinsic parameters (poses) are known.

* both parameters groups have unknown level of noise due to errors in calibration and estimating the poses.

This project is the second part of [the project here](https://github.com/engyasin/EKF-MonoSLAM_for_3D-reconstruction). This first part is EKF-MonoSLAM which gives the poses estimation and the covariances.

The camera used is caliberated in advance here ( the first program uses a virtual environement and a simulated camera and sensors, but a real one could be used in reality. However this requires changes to be made on multiple levels on the code).

If you use this project in your work, you can cite this reference here [bib]:

```
@Inbook{Yousif2021,
      author="Yousif, Yasin Maan and Hatem, Iyad",
      editor="Koubaa, Anis",
      title="Video Frames Selection Method for 3D Reconstruction Depending on ROS-Based Monocular SLAM",
      bookTitle="Robot Operating System (ROS): The Complete Reference (Volume 5)",
      year="2021",
      publisher="Springer International Publishing",
      address="Cham",
      pages="351--380",
      doi="10.1007/978-3-030-45956-7_11",
      url="https://doi.org/10.1007/978-3-030-45956-7_11"
}
```

## Usage
To get the code working ,the following requirements are needed:

* Python 2 (or 3 , but you have to change a line or two).

* Cython

* Numpy and Scipy

* OpenCV 3.0 and opencv-python package.

* [pcl for python](https://github.com/strawlab/python-pcl) for the point cloud processing.

To run the program on the data the following work is needed to be done:

- The images named by thier numbers ascendingly like *1.png , 3.png, etc..*

- The images poses formated in a way like in [this file](/Turtle_images_sim/nodes_and_prjcts.txt)

- The calibration parameters set to your values in [*read_poses_sim.py*](/read_poses_sim.py)

After that, the mode should be defined:

- If it is Sparse mode , only the important few features in every images shall be constructed with lower density and accuracy but faster than the second mode

- If the second dense mode is used, a rectangle must be drawn by the user around the object of intrest and a dense grid of features shall be taken from this region. it will be more accurate and dense but slower to run than the sparse mode.

The change between the two modes or mixing them is possible in the *main_3d.py* file by changing the different inputs (the comments showing two examples, sparse and dense).

When trying to run the code in pure python, maybe the sparse mode will be ok, but for dense mode it will take very long time (hours) for 10 to 20 images. Therefore, Cython is used here to speed up python.

The command for running the project is :

```bash
$ python main_3d.py
```

**Note**

As there are many state-of-the-art solutions to this problem, this program is not one of them. it uses a simple way in Cython and Python which can be enhanced in many ways. However,the main operations are the same as the best avaliabe solutions.

## Example Videos:

This [playlist](https://www.youtube.com/playlist?list=PLKdJ5omea_pRlrw_EUnQFm7ZJqasIBBFa) shows examples of running the two parts togther.

