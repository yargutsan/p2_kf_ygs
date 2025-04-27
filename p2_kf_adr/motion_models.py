import numpy as np

def velocity_motion_model():
    def state_transition_matrix_A():
        # TODO: Define and return the 3x3 identity matrix A
        # A should be a 3x3 identity matrix
        return np.eye(3)    
    

    def control_input_matrix_B(mu, delta_t):
        # TODO: Define B using current theta and timestep delta_t
        # B should apply linear and angular velocity to position and heading
        theta = mu[2]
        B = np.array([[delta_t * np.cos(theta), 0],
                      [delta_t * np.sin(theta), 0],
                      [0, delta_t]])
        return B

    return state_transition_matrix_A, control_input_matrix_B


def velocity_motion_model_2():
    def A():
        # TODO: Define and return the 6x6 constant velocity model transition matrix
        # A should be a 6x6 matrix
        return np.eye(6)

    def B(mu, dt):
        # TODO: Return 6x2 zero matrix (no control input used in pure KF)
        # B should be a 6x2 matrix
        B = np.zeros((6, 2))
        B[0, 0] = dt * np.cos(mu[2])
        B[1, 0] = dt * np.sin(mu[2])
        B[2, 1] = dt
        return B

    return A, B
