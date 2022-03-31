import numpy as np
from pr3_utils import *
from scipy.linalg import expm


if __name__ == '__main__':

    # Load the measurements
    filename = "/Users/orish/wi22/ece276/ECE276A_PR3/code/data/10.npz"
    t,features_all,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

    # Reduces the number of features by returning the features that appears the most in all
    # Number is selected by us 
    def reduced_ft(features,number):

        reduced_ftrs = np.zeros((features.shape[0],number,features.shape[2]))
        feature_count = np.zeros(features.shape[1])

        # Number of time each features is observed
        for t_step in range(1,np.size(t)):
            indx = np.sum(features[:,:,t_step],axis=0)!=-4
            feature_count[indx] = feature_count[indx] + 1

        # Get the number number of features observed 
        ind_reduced_ftrs = np.sort( np.flip( np.argsort(feature_count) )[:number] )
        reduced_ftrs = features[:,ind_reduced_ftrs,:]
        return reduced_ftrs

    focused_features = 5000
    features = reduced_ft(features_all,focused_features)

    M = np.zeros((4,4))
    M[:,:3] = np.vstack((K[:2,:],K[:2,:]))
    M[2,3] = -1*K[0,0]*b

    #hat-map of a 3D vector
    def hat(x):
        ans = np.array([[0,-x[2],x[1]],
                        [x[2],0,-x[0]],
                        [-x[1],x[0],0]])
        return ans

    # This  function takes linear and rotational velocities
    # reurns u hat and u curlyhat by using function: hat()
    def u_hatting(v,w):
        u_hat = np.block([[hat(w),v.reshape((3,1))],
                          [np.zeros((1,3)),0]])
        u_c_hat = np.block([[hat(w),hat(v)],
                             [np.zeros((3,3)),hat(w)]])
        return u_hat,u_c_hat


    # Function to convert features in 2D to landmarks in 3D
    def ftr2lndm(ul,vl,ur,vr):
        va = vl # averaging out the 
        #z = M[2,3]/(ur -M[2,2]-((M[2,0]/M[0,0])*(ul-M[0,2])))
        z = M[2,3]/(ur -ul)
        x = (ul-M[0,2])*z/M[0,0]
        y = (va-M[1,2])*z/M[1,1]
        
        p = np.array([x,y,z,1]).reshape(4,1)
        return p # 4x1
    

    # Function to convert landmarks from camera frame to world frame
    def lndm_in_world(pose, p):
        vector = np.matmul(pose,np.matmul(imu_T_cam,p)) # 4x1
        return vector

    ''' (b) Landmark Mapping via EKF Update '''
    
    # Initiallizations
    pose_traj = np.zeros((4,4,t.shape[1]), dtype=float)
    mu_arr = np.zeros((4,4,t.shape[1]), dtype=float)
    dmu_arr = np.zeros((6,1,t.shape[1]), dtype=float)
    var_arr = np.zeros((6,6,t.shape[1]), dtype=float)

    pixels_in_w = np.zeros((features.shape), dtype=float)

    mu0 = np.identity(4)
    var0 = np.zeros((6,6))
    dmu0 = (np.random.normal(0,var0).diagonal()).reshape((6,1))

    mu_arr[:,:,0], dmu_arr[:,:,0], var_arr[:,:,0] = mu0,dmu0,var0

    
    #co variance matrix with noise
    W = np.diag([0.01, 0.01, 0.01, 0.005, 0.005, 0.005])
    
    #co variance matrix without noise
    #W = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


    def traj_predict(ti,mu,dmu,var): #timestamp, mu of prev, del mu of prev, var prev.
        v = linear_velocity[:,ti]
        w = angular_velocity[:,ti]
        u_h, u_ch = u_hatting(v,w) # hat 4x4, curlyhat 6x6
        tau = t[0,ti+1]-t[0,ti] # scaler

        #exponent
        expp = expm((tau)*u_h) # 4x4
        cexpp = expm(-1*tau*u_ch) # 6x6

        #prediction mu, delta mu
        mu_predict = np.matmul(mu,expp) # mu(t+1/t) = 4x4
        d_mu_predict = np.matmul(cexpp,dmu) + (np.random.normal(0,W).diagonal()).reshape(6,1) # delta mu(t+1/t) = 6x1

        #prediction var
        var_predict = np.matmul(np.matmul(cexpp,var),cexpp.T) + W # 6x6

        #pose
        hat_dmu, c = u_hatting(dmu[:3,0],dmu[3:,0])
        dmu_expp = expm(hat_dmu) # 4x4
        Ti = np.matmul(mu_predict,dmu_expp) # 4x4

        return Ti,mu_predict,d_mu_predict, var_predict

    # For calculating predicted pose and variance
    # Along with the feature position in world coordinate at each time step for each feature
    def master():
        pose_traj[:,:,0] = mu0
        for ti in range(t.shape[1]-1): # loop on time
            ftrs = features[0,:,ti] # (fucus features, )
            #index of seen features
            indValid = list(np.logical_and((ftrs != -1), (ftrs != -2))) # list of boolean

            # Each pixel in world frame at each time stamp
            for px in range(features.shape[1]): # loop on features
                if indValid[px]:
                    ul,vl,ur,vr = features[:,px,ti]
                    p = ftr2lndm(ul,vl,ur,vr) # 4x1
                    wrld = lndm_in_world(pose_traj[:,:,ti], p) # 4x1 - world frame of each pixel at each time
                    pixels_in_w[:,px, ti] = wrld.reshape(4,) # filling the array
                else:
                    pixels_in_w[:,px, ti] = np.zeros((4,))
                    

            pose_traj[:,:,ti+1],mu_arr[:,:,ti+1],dmu_arr[:,:,ti+1],var_arr[:,:,ti+1] = traj_predict(ti,mu_arr[:,:,ti],dmu_arr[:,:,ti],var_arr[:,:,ti])

    master()

    # mu_underbar as stated in lecture notes
    mu_underbar = np.zeros((features.shape), dtype=float)
    # mu_underbar flat nx1
    mu_uflat = np.zeros((3*features.shape[1],1), dtype=float)
    # Calculating the average value of pixels over each time stamp - to fill mu_underbar
    def mu_und():
        for i in range(pixels_in_w.shape[1]):
            indGood = np.logical_and((pixels_in_w[0,i,:] != 0), (pixels_in_w[0,i,:] != np.inf)) # 
            count = np.count_nonzero(indGood) #scaler
            avg = np.sum(pixels_in_w[:3,i,:], axis = 1)/count
            mu_underbar[:3,i,indGood] = avg.reshape(3,1)
            mu_underbar[3,i,indGood] = 1
            mu_uflat[3*i:3*i+3,0] = avg.reshape(3,)
    mu_und()

# Initiations of mu and var of observation
mu_obs = mu_uflat
var_obs = 0.01*np.identity(3*features.shape[1])

oTi = np.linalg.inv(np.matrix(imu_T_cam))
PT = np.vstack((np.identity(3),np.array([0,0,0]).reshape(1,3)))

def dpi_by_dq(q):
    #q = q.reshape(4,)
    ans = np.identity(4)
    v = np.array([-q[0,0]/q[2,0],-q[1,0]/q[2,0],0,-q[3,0]/q[2,0]])
    ans[:,2] = v
    ans = ans/q[2,0]
    
    return ans

# Main loop
for ti in range(t.shape[1]-1):

    indGood = np.nonzero(mu_underbar[0,:,ti])[0]
    Nt = len(indGood)
    indGood = indGood.reshape(Nt,1)

    z_tilda = np.zeros((4*Nt,1))
    z = np.zeros((4*Nt,1))

    inv_T = np.linalg.inv(pose_traj[:,:,ti])  # (4x4)
    TT = np.matmul(oTi,inv_T) # 4x4

    # Second term of H
    intm2 = np.matmul(TT,PT) # (4x3)

    H = np.zeros((4*Nt,3*features.shape[1]))
    IV = np.identity(4*Nt)
    c = 0
    ch = 0
    for px in range(features.shape[1]):
        # Only the visible features at that timestamp
        if px in indGood:
            pp = np.matmul(TT,(mu_underbar[:,px,ti].reshape(4,1)))
            z_t = np.matmul(M,(pp/mu_underbar[2,px,ti]))

            # Updating z- tilda
            z_tilda[c,0] = z_t[0]
            z_tilda[c+1,0] = z_t[1]
            z_tilda[c+2,0] = z_t[2]
            z_tilda[c+3,0] = z_t[3]

            f = features[:,px,ti].reshape(4,)
            # Updating z
            z[c,0] = f[0]
            z[c+1,0] = f[1]
            z[c+2,0] = f[2]
            z[c+3,0] = f[3]
            c+=4

            temp = dpi_by_dq(pp)
            trailing = np.matmul(temp,intm2)

            # Jacobian
            H[4*ch:(4*ch)+4,3*px:(3*px)+3] = np.matmul(M,trailing)
            ch+=1

    HT = H.T
    mat = np.linalg.inv((H @ var_obs @ HT) + IV)

    # Kalman gain
    Kl = var_obs @ HT @ mat

    mu_obs = mu_obs + (Kl @ (z-z_tilda))
    var_obs = (np.eye(3*features.shape[1]) - (Kl @ H)) @ var_obs

mu_unflat = mu_obs.reshape(features.shape[1],3)
mu_x = mu_unflat[:,0].reshape(features.shape[1],1)
mu_y = mu_unflat[:,1].reshape(features.shape[1],1)

# Plotting
visualize_part_b(pose_traj,mu_x,mu_y,path_name="Trajectory",show_ori=False)



