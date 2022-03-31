import numpy as np
from pr3_utils import *
from scipy.linalg import expm


if __name__ == '__main__':

    # Load the measurements
    filename = "/Users/orish/wi22/ece276/ECE276A_PR3/code/data/10.npz"
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

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


    ''' (a) IMU Localization via EKF Prediction '''

    pose_traj = np.zeros((4,4,t.shape[1]), dtype=float)
    mu_arr = np.zeros((4,4,t.shape[1]), dtype=float)
    dmu_arr = np.zeros((6,1,t.shape[1]), dtype=float)
    var_arr = np.zeros((6,6,t.shape[1]), dtype=float)
    

    mu0 = np.identity(4)
    var0 = np.zeros((6,6))
    #delta mu0 t/t
    dmu0 = (np.random.normal(0,var0).diagonal()).reshape((6,1))

    mu_arr[:,:,0], dmu_arr[:,:,0], var_arr[:,:,0] = mu0,dmu0,var0

    
    #W - co variance matrix: zeros to account for zero noise prediction
    W = np.diag([0, 0, 0, 0, 0, 0])
    #W = np.diag([0.004, 0.004, 0.004, 0.002, 0.002, 0.002])

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

    def predict():
        pose_traj[:,:,0] = mu0
        for ti in range(t.shape[1]-1):
            pose_traj[:,:,ti+1],mu_arr[:,:,ti+1],dmu_arr[:,:,ti+1],var_arr[:,:,ti+1] = traj_predict(ti,mu_arr[:,:,ti],dmu_arr[:,:,ti],var_arr[:,:,ti])

        visualize_trajectory_2d(pose_traj,path_name="Trajectory - without noise",show_ori=True)

    predict()


