import numpy as np
import csv
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy import genfromtxt


def main():
    
    print ("first file")
    ifile=open('/home/bushra/Desktop/icp/g1.csv','r')
    lines=csv.reader(ifile)
    i=0
    flag = 0
    A = np.zeros((1,3),dtype = float)
    print(A)
    for row in lines:
        if flag == 0:
            flag = 1
            continue
        a=row
        a = np.array(a[0:3]).astype(np.float)
    
        A = np.append(A,[a[0:3]], axis = 0)
    np.array(A[1:]).astype(np.float)
    
    print(A)

    print ("second file")
    ifile1=open('/home/bushra/Desktop/icp/g2.csv','r')
    lines=csv.reader(ifile1)
    i=0
    flag = 0
    B = np.zeros((1,3),dtype = float)
    for row in lines:
        if flag == 0:
            flag = 1
            continue
        b=row
        b = np.array(b[0:3]).astype(np.float)
    
        B = np.append(B,[b[0:3]], axis = 0)
    np.array(B[1:]).astype(np.float)
    
    print(B)
    
    print(icp(B[1:],A[1:]))
    T1,d1=icp(B[1:],A[1:]) 
    
    R1=np.array(T1)[0:3,0:3]
    t1=np.array(T1)[0:3,3:]
    print("rotational vector",R1)
    print("\n")
    print("translational vector",t1)
    C=np.mat(A)*np.mat(R1)+np.mat(t1.T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    '''
    my_data = genfromtxt('f1.csv', delimiter=',')
    points1X = my_data[:,0]
    points1Y = my_data[:,1]
    points1Z = my_data[:,2]

    ## I remove the header of the CSV File.
    points1X = np.delete(points1X, 0)
    points1Y = np.delete(points1Y, 0)
    points1Z = np.delete(points1Z, 0)

    # Convert the array to 1D array
    points1X = np.reshape(points1X,points1X.size)
    points1Y = np.reshape(points1Y,points1Y.size)
    points1Z = np.reshape(points1Z,points1Z.size)
    '''

    my_data = genfromtxt('g2.csv', delimiter=',')
    points2X = my_data[:,0]
    points2Y = my_data[:,1]
    points2Z = my_data[:,2]
    ## I remove the header of the CSV File.
    points2X = np.delete(points2X, 0)
    points2Y = np.delete(points2Y, 0)
    points2Z = np.delete(points2Z, 0)

    # Convert the array to 1D array
    points2X = np.reshape(points2X,points2X.size)
    points2Y = np.reshape(points2Y,points2Y.size)
    points2Z = np.reshape(points2Z,points2Z.size)


    #ax.plot(points1X, points1Y, points1Z, 'd', markersize=8, markerfacecolor='red', label='points1')
    ax.plot(points2X, points2Y, points2Z, 'd', markersize=8, markerfacecolor='blue', label='points2')

    a = np.asarray(C)
    np.savetxt("foo.csv", a, delimiter=",")

    my_data = genfromtxt('foo.csv', delimiter=',')
    points3X = my_data[:,0]
    points3Y = my_data[:,1]
    points3Z = my_data[:,2]
    '''
    ## I remove the header of the CSV File.
    points3X = np.delete(points2X, 0)
    points3Y = np.delete(points2Y, 0)
    points3Z = np.delete(points2Z, 0)
    '''

    # Convert the array to 1D array
    points3X = np.reshape(points3X,points3X.size)
    points3Y = np.reshape(points3Y,points3Y.size)
    points3Z = np.reshape(points3Z,points3Z.size)


    # ax.plot(points2X, points2Y, points2Z, 'd', markersize=8, markerfacecolor='red', label='points1')
    ax.plot(points3X, points3Y, points3Z, 'd', markersize=8, markerfacecolor='blue', label='points2')




    plt.show()

    return
    
    
    
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    '''

    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nx3 array of points
        dst: Nx3 array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def icp(A, B, init_pose=None, max_iterations=500, tolerance=0.00000000001):
    '''
    The Iterative Closest Point method
    Input:
        A: Nx3 numpy array of source 3D points
        B: Nx3 numpy array of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation
        distances: Euclidean distances (errors) of the nearest neighbor
    '''

    # make points homogeneous, copy them so as to maintain the originals
    src = np.ones((4,A.shape[0]))
    dst = np.ones((4,B.shape[0]))
    src[0:3,:] = np.copy(A.T)
    dst[0:3,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbours between the current source and destination points
        distances, indices = nearest_neighbor(src[0:3,:].T, dst[0:3,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[0:3,:].T, dst[0:3,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.sum(distances) / distances.size
        if abs(prev_error-mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[0:3,:].T)
    

    return T, distances

if __name__ == "__main__":
    main()
    pass
