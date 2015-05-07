'''
Created on Nov 25, 2014

@author: kasimsert
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import manifold
import codecs


def lle(data,nRedDim=2,K=30):

    ndata = np.shape(data)[0]
    ndim = np.shape(data)[1]
    d = np.zeros((ndata,ndata),dtype=float)
    
    # Inefficient -- not matrices
    for i in range(ndata):
        for j in range(i+1,ndata):
            for k in range(ndim):
                d[i,j] += (data[i,k] - data[j,k])**2
            d[i,j] = np.sqrt(d[i,j])
            d[j,i] = d[i,j]

    indices = d.argsort(axis=1)
    neighbours = indices[:,1:K+1]

    W = np.zeros((K,ndata),dtype=float)

    for i in range(ndata):
        Z  = data[neighbours[i,:],:] - np.kron(np.ones((K,1)),data[i,:])
        C = np.dot(Z,np.transpose(Z))
        C = C+np.identity(K)*1e-3*np.trace(C)#This ensures that the system to be solved in step 2 has a unique solution.
        W[:,i] = np.transpose(np.linalg.solve(C,np.ones((K,1))))
        W[:,i] = W[:,i]/np.sum(W[:,i])

    M = np.eye(ndata,dtype=float)
    for i in range(ndata):
        w = np.transpose(np.ones((1,np.shape(W)[0]))*np.transpose(W[:,i]))
        j = neighbours[i,:]
        #print shape(w), np.shape(np.dot(w,np.transpose(w))), np.shape(M[i,j])
        ww = np.dot(w,np.transpose(w))
        for k in range(K):
            M[i,j[k]] -= w[k]
            M[j[k],i] -= w[k]
            for l in range(K):
                M[j[k],j[l]] += ww[k,l]
    
    evals,evecs = np.linalg.eig(M)
    ind = np.argsort(evals)
    y = evecs[:,ind[1:nRedDim+1]]*np.sqrt(ndata)
    return evals,evecs,y

reader = csv.reader(codecs.open("/Users/kasimsert/Downloads/ilmesafe2.csv", "r",encoding='iso8859-9'), delimiter=';')
data = list(reader)

dists = []
cities = []
for d in data:
    cities.append(d[0])
    dists.append(map(float , d[1:-1]))


adist = np.array(dists)
amax = np.amax(adist)
adist /= amax

evals,evecs,coords = lle(adist)

plt.subplots_adjust(bottom = 0.1)
# plt.scatter(coords[:, 0] , coords[:, 1], marker = 'o')
# for label, x, y in zip(cities, coords[:, 0], coords[:, 1]):
plt.scatter(coords[:, 0]*-1 , coords[:, 1]*-1, marker = 'o')
for label, x, y in zip(cities, coords[:, 0]*-1, coords[:, 1]*-1):
    plt.annotate(
        label,
        xy = (x, y), xytext = (-10, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.show()
