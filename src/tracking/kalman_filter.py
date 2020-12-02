import numpy as np


class KalmanFilter:
    
    def __init__(self, H, x0, A, Q, R, P0=None):
        if H.shape[1] != x0.shape[0] or A.shape[0] != x0.shape[0]:
            raise ValueError('Incompatible matrix shapes')        
        self.A = A
        self.Q = Q
        self.x0 = x0
        self.x_k = x0
        self.H = H
        if P0 is None:
            P0 = 4*Q
            
        self.P0 = P0
        self.P_k = P0
        self.k = 0
        self.R = R
        self.I = np.eye(x0.shape[0])
        
    def one_step(self, x=None, P=None):
        if x is None:
            x = self.x_k
        if P is None:
            P = self.P_k
        
        # A priori state evolution, 1-step observation prediction, state covariance
        x_k = np.matmul(self.A, x)
        y_k = np.matmul(self.H, x_k)
        P_k = np.matmul(np.matmul(self.A, P), self.A.transpose()) + self.Q
        
        return x_k, y_k, P_k

    def multi_step(self, steps, x=None, P=None):
        if x is None:
            x = self.x_k
        if P is None:
            P = self.P_k
            
        x_all = np.zeros((steps, x.shape[0]))
        y_all = np.zeros((steps, self.H.shape[0]))
        
        for i in range(0, steps):
            x_k, y_k, P_k = self.one_step(x, P)
            
            x_all[i, :] = x_k
            y_all[i, :] = y_k
            
            x = x_k
            P = P_k
            
        return x_all, y_all
        

    def predict(self, steps=1):
        
        P_k = self.P_k
        x_k = self.x
        y = np.matmul(H,x)
        
        prj = list()
        
        for i in np.arange(0, steps):

            x_k, y_k, P_k = self.one_step(x_k, P_k)
            
            prj_k = {'x_k': x_k,
                     'y_k': y_k,
                     'P_k': P_k}
            
            prj.append(prj_k)
        
        return prj
            
    def update(self, y):
        """
        Perform a 1-step prediction
        """
        
        x_k, y_k, P_k = self.one_step()
        
        # One-step prediction error
        err = y - y_k
        
        S_k = np.matmul(np.matmul(self.H, self.P_k), self.H.transpose()) + self.R
        
        # Kalman gain
        K = np.matmul(np.matmul(P_k, self.H.transpose()), np.linalg.inv(S_k))
        
        # Posterior update from observation error
        self.x_k = x_k + np.matmul(K, err)
        self.P_k = np.matmul((self.I - np.matmul(K, self.H)), P_k)
        
        return y_k, S_k
    
    def advance_no_observation(self):
        """
        Advance state (forward predict) without the benefit of observation updates for
        interpolation or forward prediction (no posterior updates).  

        Note that this method updates the internal state without observation updates - if instead 
        you want to forward-predict (but retain the current state for subsequent updates), use the 
        one_step() or multi_step() methods which generate forward predictions without changing the 
        filter's internal state.   
        """
        x_k, y_k, P_k = self.one_step
        
        self.x = x_k
        self.P = P_k
        
        return x_k, y_k, P_k

    @classmethod
    def init_2dtracker(cls, x=0., y=0., v_decay=0.98, 
                       accel_decay=0.98, obs_cov=0.0004):

        # observables: x&y coordinates
        # internal states: x, y, dx/dt, dy/dt, d2x/dt2, d2y/dt2

        vd = v_decay
        ad = v_decay

        x0 = np.array([x,  y,  0., 0., 0., 0.])
        A = np.array([[1., 0., 1., 0., 0., 0.],
                      [0., 1., 0., 1., 0., 0.],
                      [0., 0., vd, 0., 1., 0.],
                      [0., 0., 0., vd, 0., 1.],
                      [0., 0., 0., 0., ad, 0.],
                      [0., 0., 0., 0., 0., ad]])

        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]])

        Q = np.diag([1e-4, 1e-4, 1e-6, 1e-6, 4e-6, 4e-6])
        R = np.diag([obs_cov, obs_cov])

        kf = KalmanFilter(H=H, x0=x0, A=A, Q=Q, R=R)
        return kf
