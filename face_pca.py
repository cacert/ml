from PIL import Image
from numpy import *
import pylab
 
def pca(X):
    # Principal Component Analysis
    # input: X, matrix with training data as flattened arrays in rows
    # return: projection matrix (with important dimensions first),
    # variance and mean
   
    # get dimensions
    num_data, dim = X.shape
   
    # center data
    mean_X = X.mean(axis=0)
    for i in range(num_data):
        X[i] -= mean_X
   
    COVARIANCE_MATRIX = dot(X, X.T)  # covariance matrix
    e, EV = linalg.eigh(COVARIANCE_MATRIX)  # eigenvalues and eigenvectors
    tmp = dot(X.T, EV).T  # this is the compact trick
    V = tmp[::-1]  # reverse since last eigenvectors are the ones we want
    S = sqrt(e)[::-1]  # reverse since eigenvalues are in increasing order
   
    # return the projection matrix, the variance and the mean
    return V, S, mean_X
 
 
def loadImage():
    imlist=[]
    imlist.append("gwb_cropped/George_W_Bush_0001.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0006.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0009.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0015.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0025.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0027.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0031.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0035.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0043.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0049.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0057.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0066.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0068.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0072.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0078.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0088.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0089.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0091.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0105.jpg")
    imlist.append("gwb_cropped/George_W_Bush_0113.jpg")
    return imlist

from PIL import Image
import numpy
import pylab
 
imlist = loadImage()
im = numpy.array(Image.open(imlist[0]))  # open one image to get the size
m, n = im.shape[0:2]  # get the size of the images
imnbr = len(imlist)  # get the number of images
 
# create matrix to store all flattened images
immatrix = numpy.array([numpy.array(Image.open(imlist[i])).flatten() for i in range(imnbr)], 'f')
 
# perform PCA
V, S, immean = pca(immatrix)
 
# mean image and first mode of variation
immean = immean.reshape(m, n)
mode = V[0].reshape(m, n)
 
# show the images
pylab.figure()
pylab.gray()
pylab.imshow(immean)
 
pylab.figure()
pylab.gray()
pylab.imshow(mode)
 
pylab.show()