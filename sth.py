import numpy as np
cimport numpy as np
import cv2

cimport cython

ctypedef unsigned char SMALL_INT
ctypedef np.float32_t FLOAT

cdef extern from "math.h":
    double sqrt(double m)

cdef np.ndarray[FLOAT, ndim=2] gradient(np.ndarray[SMALL_INT, ndim=2] img, char dxdy):
    cdef np.ndarray[FLOAT, ndim=2] d
    d = cv2.Sobel(img,  cv2.CV_32F, dxdy==0, dxdy==1,  ksize = 1,  delta = 0.5,  scale = 0.01)
    return d

@cython.boundscheck(False)
def swt(np.ndarray[SMALL_INT, ndim=3] img, np.ndarray[SMALL_INT, ndim=2] mask, int thStart, int thStop, int pDivider, int lMin, int lMax):
    cdef np.ndarray[SMALL_INT, ndim=3] outImg
    cdef np.ndarray[FLOAT, ndim=2] dRow, dCol, gradMag
    cdef np.ndarray[SMALL_INT, ndim=2] imgGray, dRowUint, dRowPosMask, startSWT, stopSWT, gradMagUint
    cdef int h,w,ch, points, row, col, i, l, r, c
    cdef float dirRow, dirCol, dirMag
    cdef np.ndarray[long, ndim=1] startPointsRow, startPointsCol        

    t1 = cv2.getTickCount();    

    outImg = img
    h,  w,  ch = np.shape(img)

    # Calculate gradients
    imgGray = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
    dCol = gradient(imgGray, 0)
    dRow = gradient(imgGray, 1)

    # Make a mask for pos row-gradient (This is startpoint in the SWT)
    dRowUint = cv2.convertScaleAbs(dRow,  alpha=255)
    ret,  dRowPosMask = cv2.threshold(dRowUint, 128, 255, cv2.THRESH_BINARY)

    # Calculate magnitude of gradients
    gradMag = cv2.absdiff(dRow, 0.5) + cv2.absdiff(dCol, 0.5)
    gradMagUint = cv2.convertScaleAbs(gradMag,  alpha=255)

    # Find suitable startpoints for the SWT
    startSWT = np.bitwise_and(gradMagUint, dRowPosMask)
    startSWT = np.bitwise_and(startSWT,  mask)
    ret,  startSWT = cv2.threshold(startSWT,  thStart,  255,  cv2.THRESH_TOZERO)
    startPointsRow,  startPointsCol = np.nonzero(startSWT)

    # Find stop points
    stopSWT = np.bitwise_and(gradMagUint, np.invert(dRowPosMask))
    stopSWT = np.bitwise_and(stopSWT,  mask)

    pointsTuple = np.shape(startPointsRow)
    points = pointsTuple[0];

    t2 = cv2.getTickCount();

    # Step until stopSWT > th
    for i in range(points):
        # Find start pos
        row = startPointsRow[i]
        col = startPointsCol[i]
        # Find direction
        dirRow = -dRow[row][col]+0.5
        dirCol = -dCol[row][col]+0.5
        dirMag = sqrt(dirRow*dirRow + dirCol*dirCol)
        dirRow /= dirMag
        dirCol /= dirMag
        # Step until stop found or l > 100
        l = 1
        r = <int>(row + l*dirRow)
        c = <int>(col + l*dirCol)
        while(r<h and r>=0 and c<w and c>=0 and stopSWT[r][c] < thStop and l < lMax):
            l  += 1;
            r = <int>(row + l*dirRow)
            c = <int>(col + l*dirCol)
        if (r<h and r>=0 and c<w and c>=0 and l < lMax and l > lMin):
            cv2.line(outImg,  (col, row),  (c, r), (0, 255, 0))
        elif (l > lMax):
            cv2.line(outImg,  (col, row),  (c, r), (255, 0, 0))
        elif (l < lMin):
            cv2.line(outImg,  (col, row),  (c, r), (0, 0, 255))

    t3 = cv2.getTickCount();

    print((t2-t1)/cv2.getTickFrequency())
    print((t3-t2)/cv2.getTickFrequency())        

    return outImg








from multiprocessing import Pool
import time, random, sys

#Dependencies defined below main()

def main():
    """
    This is the main method, where we:
    -generate a random list.
    -time a sequential mergesort on the list.
    -time a parallel mergesort on the list.
    -time Python's built-in sorted on the list.
    """
    N = 500000
    if len(sys.argv) > 1:  #the user input a list size.
        N = int(sys.argv[1])

    #We want to sort the same list, so make a backup.
    lystbck = [random.random() for x in range(N)]

    #Sequential mergesort a copy of the list.
    lyst = list(lystbck)
    start = time.time()             #start time
    lyst = mergesort(lyst)
    elapsed = time.time() - start   #stop time

    if not isSorted(lyst):
        print('Sequential mergesort did not sort. oops.')
    
    print('Sequential mergesort: %f sec' % (elapsed))


    #So that cpu usage shows a lull.
    time.sleep(3)


    #Now, parallel mergesort. 
    lyst = list(lystbck)
    start = time.time()
    n = 3 #2**(n+1) - 1 processes will be instantiated.

    #Instantiate a Process and send it the entire list,
    #along with a Pipe so that we can receive its response.
    lyst = mergeSortParallel(lyst, n)

    elapsed = time.time() - start

    if not isSorted(lyst):
        print('mergeSortParallel did not sort. oops.')

    print('Parallel mergesort: %f sec' % (elapsed))


    time.sleep(3)
    
    #Built-in test.
    #The underlying c code is obviously the fastest, but then
    #using a calculator is usually faster too.  That isn't the
    #point here obviously.
    lyst = list(lystbck)
    start = time.time()
    lyst = sorted(lyst)
    elapsed = time.time() - start
    print('Built-in sorted: %f sec' % (elapsed))


def merge(left, right):
    """returns a merged and sorted version of the two already-sorted lists."""
    ret = []
    li = ri = 0
    while li < len(left) and ri < len(right):
        if left[li] <= right[ri]:
            ret.append(left[li])
            li += 1
        else:
            ret.append(right[ri])
            ri += 1
    if li == len(left):
        ret.extend(right[ri:])
    else:
        ret.extend(left[li:])
    return ret

def mergesort(lyst):
    """
    The seemingly magical mergesort. Returns a sorted copy of lyst.
    Note this does not change the argument lyst.
    """
    if len(lyst) <= 1:
        return lyst
    ind = len(lyst)//2
    return merge(mergesort(lyst[:ind]), mergesort(lyst[ind:]))

def mergeWrap(AandB):
    a,b = AandB
    return merge(a,b)

def mergeSortParallel(lyst, n):
    """
    Attempt to get parallel mergesort faster in Windows.  There is
    something wrong with having one Process instantiate another.
    Looking at speedup.py, we get speedup by instantiating all the
    processes at the same level. 
    """
    numproc = 2**n
    #Evenly divide the lyst indices.
    endpoints = [int(x) for x in linspace(0, len(lyst), numproc+1)]
    #partition the lyst.
    args = [lyst[endpoints[i]:endpoints[i+1]] for i in range(numproc)]

	#instantiate a Pool of workers
    pool = Pool(processes = numproc)
    sortedsublists = pool.map(mergesort, args)
	#i.e., perform mergesort on the first 1/numproc of the lyst, 
	#the second 1/numproc of the lyst, etc.

    #Now we have a bunch of sorted sublists.  while there is more than
    #one, combine them with merge.
    while len(sortedsublists) > 1:
        #get sorted sublist pairs to send to merge
        args = [(sortedsublists[i], sortedsublists[i+1]) \
				for i in range(0, len(sortedsublists), 2)]
        sortedsublists = pool.map(mergeWrap, args)

	#Since we start with numproc a power of two, there will always be an 
	#even number of sorted sublists to pair up, until there is only one.

    return sortedsublists[0]
    

    
def linspace(a,b,nsteps):
    """
    returns list of simple linear steps from a to b in nsteps.
    """
    ssize = float(b-a)/(nsteps-1)
    return [a + i*ssize for i in range(nsteps)]


def isSorted(lyst):
    """
    Return whether the argument lyst is in non-decreasing order.
    """
    #Cute list comprehension way that doesn't short-circuit.
    #return len([x for x in
    #            [a - b for a,b in zip(lyst[1:], lyst[0:-1])]
    #            if x < 0]) == 0
    for i in range(1, len(lyst)):
        if lyst[i] < lyst[i-1]:
            return False
    return True

#Execute the main method now that all the dependencies
#have been defined.
#The if __name is so that pydoc works and we can still run
#on the command line.
if __name__ == '__main__':
    main()
