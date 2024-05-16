
import numpy as np  # Built by me

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise, measurement_matrix):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.measurement_matrix = measurement_matrix

    def predict(self, control_matrix, control_vector):
        # Predict the next state
        self.state = np.dot(control_matrix, self.state) + control_vector
        self.covariance = np.dot(np.dot(control_matrix, self.covariance), control_matrix.T) + self.process_noise

        # Check for NaN and reset if necessary
        if np.isnan(self.state).any() or np.isnan(self.covariance).any():
            print("NaN detected in prediction step, resetting...")
            self.state = np.zeros_like(self.state)  # Reset state to zero or another suitable default
            self.covariance = np.eye(len(self.state)) * 1000  # Large initial covariance

    def update(self, measurement):
        # Compute the Kalman Gain
        S = np.dot(self.measurement_matrix, np.dot(self.covariance, self.measurement_matrix.T)) + self.measurement_noise
        if np.linalg.cond(S) < 1 / np.finfo(float).eps:
            K = np.dot(np.dot(self.covariance, self.measurement_matrix.T), np.linalg.inv(S))
        else:
            print("Condition number too high, skipping update...")
            return  # Skip update if S is near singular

        # Update the state estimate
        y = measurement - np.dot(self.measurement_matrix, self.state)
        self.state += np.dot(K, y)
        self.covariance = (np.eye(self.state.shape[0]) - np.dot(K, self.measurement_matrix)) * self.covariance

        # Adapt process noise based on residual magnitude
        self.adapt_process_noise(y)

        # Check for NaN and reset if necessary
        if np.isnan(self.state).any() or np.isnan(self.covariance).any():
            print("NaN detected in update step, resetting...")
            self.state = np.zeros_like(self.state)
            self.covariance = np.eye(len(self.state)) * 1000

    def adapt_process_noise(self, residual):
        residual_norm = np.linalg.norm(residual)
        threshold_high = 10.0  # Set your threshold value
        threshold_low = 1.0  # Set your threshold value

        if residual_norm > threshold_high:
            self.process_noise *= 1.1  # Increase process noise
        elif residual_norm < threshold_low:
            self.process_noise *= 0.9  # Decrease process noise

        # Ensure process noise does not become too low or too high
        self.process_noise = np.clip(self.process_noise, 0.01, 1000)