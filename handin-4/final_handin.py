# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:01:45 2017

@author: AU
"""
import os
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Load the Iris data set
import sklearn.datasets
X, y = sklearn.datasets.load_iris(True)
X = X[:,0:2] # reduce to 2d so you can plot if you want
print(X.shape, y.shape)

def lloyds_algorithm(X, k, T):
    """ Clusters the data of X into k clusters using T iterations of Lloyd's algorithm. 
    
        Parameters
        ----------
        X : Data matrix of shape (n, d)
        k : Number of clusters.
        T : Maximum number of iterations to run Lloyd's algorithm. 
        
        Returns
        -------
        clustering: A vector of shape (n, ) where the i'th entry holds the cluster of X[i].
        centroids:  The centroids/average points of each cluster. 
        cost:       The cost of the clustering 
    """
    n, d = X.shape
    
    # Initialize clusters random. 
    clustering = np.random.randint(0, k, (n, )) 
    centroids  = np.zeros((k, d))
    
    # Used to stop if cost isn't improving (decreasing)
    cost = 0
    oldcost = 0
    
    # Column names
    #print("Iterations\tCost")
    
    for i in range(T):
        # Update centroid
        # YOUR CODE HERE
        for cent in range (k):
            if len(X[clustering==cent,])!=0:
                centroids[cent,]=sum(X[clustering==cent,])/len(X[clustering==cent,])
        # END CODE

        # Update clustering 
        # YOUR CODE HERE
        for point in range(n):
            dist = np.zeros(k)
            for cluster in range(k):
#                dist[cluster]=sum(np.square(X[point,]-centroids[cluster,]))
                dist[cluster]=np.linalg.norm(X[point] - centroids[cluster])**2
            clustering[point]=np.argmin(dist)
        # END CODE

        # Compute and print cost
        cost = 0
        for j in range(n):
            cost += np.linalg.norm(X[j] - centroids[clustering[j]])**2    
        #print(i+1, "\t\t", cost)
        
        # Stop if cost didn't improve more than epislon (decrease)
        if np.isclose(cost, oldcost): break #TODO
        oldcost = cost
        
    return clustering, centroids, cost

clustering, centroids, cost = lloyds_algorithm(X, 3, 100)


def compute_probs_cx(points, means, covs, probs_c):
    '''
    Input
      - points: (n times d) array containing the dataset
      - means:  (k times d) array containing the k means
      - covs:   (k times d times d) array such that cov[j,:,:] is the covariance matrix of the j-th Gaussian.
      - priors: (k) array containing priors
    Output
      - probs:  (k times n) array such that the entry (i,j) represents Pr(C_i|x_j)
    '''
    # Convert to numpy arrays.
    points, means, covs, probs_c = np.asarray(points), np.asarray(means), np.asarray(covs), np.asarray(probs_c)
    
    # Get sizes
    n, d = points.shape
    k = means.shape[0]
    
    # Compute probabilities
    # This will be a (k, n) matrix where the (i,j)'th entry is Pr(C_i)*Pr(x_j|C_i).
    probs_cx = np.zeros((k, n))
    for i in range(k):
        try:
            probs_cx[i] = probs_c[i] * multivariate_normal.pdf(mean=means[i], cov=covs[i], x=points)
        except Exception as e:
            print("Cov matrix got singular: ", e)
    
    # The sum of the j'th column of this matrix is P(x_j); why?
    probs_x = np.sum(probs_cx, axis=0, keepdims=True) 
    assert probs_x.shape == (1, n)
    
    # Divide the j'th column by P(x_j). The the (i,j)'th then 
    # becomes Pr(C_i)*Pr(x_j)|C_i)/Pr(x_j) = Pr(C_i|x_j)
    probs_cx = probs_cx / probs_x
    
    return probs_cx, probs_x
    
    
def em_algorithm(X, k, T, epsilon = 0.001, means=None):
    """ Clusters the data X into k clusters using the Expectation Maximization algorithm. 
    
        Parameters
        ----------
        X : Data matrix of shape (n, d)
        k : Number of clusters.
        T : Maximum number of iterations
        epsilon :  Stopping criteria for the EM algorithm. Stops if the means of
                   two consequtive iterations are less than epsilon.
        means : (k times d) array containing the k initial means (optional)
        
        Returns
        -------
        means:     (k, d) array containing the k means
        covs:      (k, d, d) array such that cov[j,:,:] is the covariance matrix of 
                   the Gaussian of the j-th cluster
        probs_c:   (k, ) containing the probability Pr[C_i] for i=0,...,k. 
        llh:       The log-likelihood of the clustering (this is the objective we want to maximize)
    """
    n, d = X.shape
    
    # Initialize and validate mean
    if means is None: 
        means = np.random.rand(k, d)

    # Initialize cov, prior
    probs_x  = np.zeros(n) 
    probs_cx = np.zeros((k, n)) 
    probs_c  = np.zeros(k) + np.random.rand(k)
    
    covs = np.zeros((k, d, d))
    for i in range(k): covs[i] = np.identity(d)
    probs_c = np.ones(k) / k
    
    # Column names
    #print("Iterations\tLLH")
    
    close = False
    old_means = np.zeros_like(means)
    iterations = 0
    while not(close) and iterations < T:
        old_means[:] = means 

        # Expectation step
        probs_cx, probs_x = compute_probs_cx(X, means, covs, probs_c)
        assert probs_cx.shape == (k, n)
        
        # Maximization step
        # YOUR CODE HERE
#        print(probs_c)
        for i in range(k):
#            count = 0;
#            for j in range(n):
#                if np.argmax(probs_cx[:,j])==i:
#                    probs_c[i] += probs_cx[i,j]
#                    count+=1
#            if count!=0:
#                probs_c[i]=probs_c[i]/count 
            probs_c[i] = np.sum(probs_cx[i])/n
            #print("probs_c : ",probs_c)
            
            means[i] = np.dot(probs_cx[i],X)/np.sum(probs_cx[i])
            #print("means : ",means)
            
#            weighted_sum = sum([probs_cx[i,j] * np.outer(X[j] - means[i],X[j] - means[i]) for j in range(n)])
            weighted_sum = sum(probs_cx[i,j] * np.outer(X[j] - means[i],X[j] - means[i]) for j in range(n))
            denom = np.sum(probs_cx[i])
            
            covs[i] = np.array(weighted_sum/denom)
            
            #print("covariance : ",covs)
        
        # END CODE
        
        # Compute per-sample average log likelihood (llh) of this iteration     
        llh = 1/n*np.sum(np.log(probs_x))
        #print(iterations+1, "\t\t", llh)

        # Stop condition
        dist = np.sqrt(((means - old_means) ** 2).sum(axis=1))
        close = np.all(dist < epsilon)
        iterations += 1
        
    # Validate output
    assert means.shape == (k, d)
    assert covs.shape == (k, d, d)
    assert probs_c.shape == (k,)
    
    return means, covs, probs_c, llh


means, covs, probs, llh = em_algorithm(X, 3, 10, epsilon = 0.001, means = centroids) 
   
def compute_em_clustering(means, covs, probs_c, data):
    probs_cx, probs_c = compute_probs_cx(data, means, covs, probs_c)
    clustering = probs_cx.argmax(axis=0)
    return clustering

clustering = compute_em_clustering(means, covs, probs, X)    
f1(clustering, y)
    
    
### 2.1 Silhouette Coefficient
def silhouette(data, clustering): 
    n, d = data.shape
    k = np.unique(clustering)[-1]+1

    # YOUR CODE HERE
    silh = np.zeros(n)

    for point in range(n):
        cohesion = 0
        separation = []
        for j in range(n):
            if clustering[point]==clustering[j] and point!=j:
                cohesion += np.linalg.norm(data[point]-data[j])
        cohesion = cohesion/(np.count_nonzero(clustering==clustering[point])-1)
        for cent in range(k):
            if cent != clustering[point]:
                s = 0
                for j in range(n):
                    if clustering[j] == cent:
                        s+=np.linalg.norm(data[point]-data[j])
                separation.append(s)
        silh[point]=(min(separation)-cohesion)/max(min(separation),cohesion)
    # END CODE

    return silh
    
np.mean(silhouette(X, clustering))

    
### Test clusterings using Silhouette
for k in range(2, 10):
    
    clustering, centroids, cost = lloyds_algorithm(X, k, 50)
    lloyd_sc = np.mean(silhouette(X, clustering))
    print("LLoyd:", k, lloyd_sc)

    means, covs, probs, llh = em_algorithm(X, k, 100, epsilon = 0.001, means = centroids)    
    clustering = compute_em_clustering(means, covs, probs, X)    
    em_score = np.mean(silhouette(X,clustering))
    print("EM:", k, em_score)
    
    # (Optional) try the lloyd's initialized EM algorithm.    


### 2.1 F1 Score
def f1(predicted, labels):
    n, = predicted.shape
    assert labels.shape == (n,)
    r = np.max(predicted) + 1
    k = np.max(labels) + 1

    # Implement the F1 score here
    # YOUR CODE HERE
    contingency = np.zeros((r,k))
    F_individual = np.zeros(r)
    F_overall = None
    for i in range(n):
        contingency[predicted[i],labels[i]]+=1
    precision = np.zeros(r)
    recall = np.zeros(r)
    for i in range(r):
        precision[i] = np.max(contingency[i,])/np.sum(contingency[i,])
        recall[i] = np.max(contingency[i,])/np.sum(contingency[:,np.argmax(contingency[i])])
        F_individual[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])
#        f = 2*np.max(contingency[i,])/(np.sum(contingency[i,])+np.sum(contingency[:,np.argmax(contingency[i])]))
#       print(F_individual, ' = ', f)
    F_overall = np.mean(F_individual)
    # END CODE

    assert contingency.shape == (r, k)
    return F_individual, F_overall, contingency    

f1(clustering,y)
    
### Test clusterings using F1 Score
for k in range(2, 10):
    #... = em_algorithm(X, k, 50)
    # em_sc = f1(...)
    #print(k, em_sc)
    
    clustering, centroids, cost = lloyds_algorithm(X, k, 50)
    lloyd_sc = f1(clustering,y)
    print(k, lloyd_sc[1])

    means, covs, probs, llh = em_algorithm(X, k, 100, epsilon = 0.001, means = centroids)    
    clustering = compute_em_clustering(means, covs, probs, X)    
    em_sc = f1(clustering,y)
    print("EM:", k, em_sc[1])
    
    # (optional) Try the lloyd's initialized EM algorithm.


    
### 3. Compressing images
import scipy.misc
import matplotlib.pyplot as plt
import os

def download_image(url):
    filename = url[url.rindex('/')+1:]
    try:
        with open(filename, 'rb') as fp:
            return scipy.misc.imread(fp) / 255
    except FileNotFoundError:
        import urllib.request
        with open(filename, 'w+b') as fp, urllib.request.urlopen(url) as r:
            fp.write(r.read())
            return scipy.misc.imread(fp) / 255
 
img_facade = download_image('https://uploads.toptal.io/blog/image/443/toptal-blog-image-1407508081138.png')

fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.imshow(img_facade)
plt.show()

size = os.stat('toptal-blog-image-1407508081138.png').st_size

print("The image consumes a total of %i bytes. \n"%size)
print("You should compress your image as much as possible! ")

def compress_kmeans(im, k, T, name):
    height, width, depth = im.shape
    data = im.reshape((height * width, depth))
    clustering, centroids, score = lloyds_algorithm(data, k, 5)
    
    # make each entry of data to the value of it's cluster
    data_compressed = data
    
    for i in range(k): data_compressed[clustering == i] = centroids[i] 
    
    im_compressed = data_compressed.reshape((height, width, depth))
    
    # The following code should not be changed. 
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(im_compressed)
    plt.savefig("compressed.jpg")
    plt.show()
    
    original_size   = os.stat(name).st_size
    compressed_size = os.stat('compressed.jpg').st_size
    print("Original Size: \t\t", original_size)
    print("Compressed Size: \t", compressed_size)
    print("Compression Ratio: \t", round(original_size/compressed_size, 5))

def compress_facade(k=8, T=100):
    img_facade = download_image('https://users-cs.au.dk/rav/ml/handins/h4/nygaard_facade.jpg')
    compress_kmeans(img_facade, k, T, 'nygaard_facade.jpg')
    
compress_facade()