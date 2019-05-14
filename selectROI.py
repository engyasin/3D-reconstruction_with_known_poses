from __future__ import print_function

"""
Do a mouseclick somewhere, move the mouse to some destination, release
the button.  This class gives click- and release-events and also draws
a line or a box from the click-point to the actual mouseposition
(within the same axes) until the button is released.  Within the
method 'self.ignore()' it is checked whether the button from eventpress
and eventrelease are the same.

"""
from matplotlib.widgets import RectangleSelector
import numpy as np
import cv2
import matplotlib.pyplot as plt

def ts2(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and ts2.RS.active:
        print(' RectangleSelector deactivated.')
        ts2.RS.set_active(False)
    if event.key in ['A', 'a'] and not ts2.RS.active:
        print(' RectangleSelector activated.')
        ts2.RS.set_active(True)


class SelectROI():

    def __init__(self, image_one = './Turtle_images_sim/2.png'):
        fig, current_ax = plt.subplots()                 # make a new plotting range
        N = 100000                                       # If N is large one can see
        x = np.linspace(0.0, 10.0, N)                    # improvement by use blitting!

        #cv2.namedWindow('Choose a rectangle around the object of intrest')

        img = cv2.imread(image_one)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #cv2.imshow('Choose a rectangle around the object of intrest' , img )
        plt.imshow(img)
        #plt.show()

        print("\n      click  -->  release")

        # drawtype is 'box' or 'line' or 'none'
        self.toggle_selector0 = ts2
        self.toggle_selector0.RS = RectangleSelector(current_ax, self.line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels')
        plt.connect('key_press_event', self.toggle_selector0)
        plt.show()
    

    def line_select_callback(self,eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        self.rect = [x1,y1,x2,y2]
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))








