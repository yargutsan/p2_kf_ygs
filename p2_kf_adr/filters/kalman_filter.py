import numpy as np 

from ..motion_models import velocity_motion_model, velocity_motion_model_2
from ..observation_models import odometry_observation_model, odometry_observation_model_2

class KalmanFilter:

    def __init__(self, initial_state, initial_covariance, proc_noise_std = [0.5, 0.5, 0.3], obs_noise_std = [0.02, 0.02, 0.01]):
        self.mu = initial_state # Initial state estimate [x, y, theta]
        self.Sigma = initial_covariance # Initial uncertainty

        self.A, self.B = velocity_motion_model() # The action model to use. Returns A and B matrices

        # Standard deviations for the noise in x, y, and theta (process or action model noise)
        self.proc_noise_std = np.array(proc_noise_std)
        # Process noise covariance (R)
        self.R = np.diag(self.proc_noise_std ** 2)  # process noise covariance

        # Observation model (C)
        self.C = odometry_observation_model() # The observation model to use

        # Standard deviations for the noise in x, y, theta (observation or sensor model noise)
        self.obs_noise_std = np.array(obs_noise_std)
        # Observation noise covariance (Q)
        self.Q = np.diag(self.obs_noise_std ** 2)
            
    def predict(self, u, dt):
        # TODO: Implement Kalman filter prediction step
        # Predict the new mean (mu) using A, B, and control input u
        # Predict the new covariance (Sigma) using A and R
        A = self.A()
        B = self.B(self.mu, dt)

        self.mu = A @ self.mu + B @ u
        self.Sigma = A @ self.Sigma @ A.T + self.R

        return 

    def update(self, z):
        # TODO: Implement Kalman filter correction step
        # Compute Kalman gain K
        # Update the mean (mu) with the measurement z
        # Update the covariance (Sigma)

        K = self.Sigma @ self.C.T @ np.linalg.inv(self.C @ self.Sigma @ self.C.T + self.Q)
        self.mu = self.mu + K @ (z - self.C @ self.mu)
        self.Sigma = (np.eye(len(self.mu)) - K @ self.C) @ self.Sigma  

        return

class KalmanFilter_2:
    def __init__(self, initial_state, initial_covariance,
                 proc_noise_std=[0.02]*6, obs_noise_std=[0.02]*6):

        self.mu = initial_state  # Initial state estimate [x, y, theta, vx, vy, omega]
        self.Sigma = initial_covariance  # Initial uncertainty

        self.A, self.B = velocity_motion_model_2()  # Motion model matrices

        self.proc_noise_std = np.array(proc_noise_std)
        self.R = np.diag(self.proc_noise_std ** 2)  # Process noise covariance

        self.C = odometry_observation_model_2()  # Observation matrix
        self.obs_noise_std = np.array(obs_noise_std)
        self.Q = np.diag(self.obs_noise_std ** 2)  # Observation noise covariance

    def predict(self, u=None, dt=1.0):
        # TODO: Implement Kalman prediction step for full state (6D)
        # Pure KF: use only the A matrix to update the state and covariance
        A = self.A()
        B = self.B(self.mu, dt)
        self.mu = A @ self.mu + B @ u if u is not None else self.mu
        self.Sigma = A @ self.Sigma @ A.T + self.R

    def update(self, z):
        # TODO: Implement update step
        # Compute Kalman gain
        # Correct the predicted state with measurement
        # Update covariance
        C = self.C
        K = self.Sigma @ C.T @ np.linalg.inv(C @ self.Sigma @ C.T + self.Q)
        self.mu = self.mu + K @ (z - C @ self.mu)
        self.Sigma = (np.eye(len(self.mu)) - K @ C) @ self.Sigma
