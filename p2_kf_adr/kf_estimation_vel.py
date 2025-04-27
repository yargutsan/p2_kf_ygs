import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

import numpy as np
import math

from .sensor_utils import odom_to_pose2D, get_normalized_pose2D, generate_noisy_measurement_2, Odom2DDriftSimulator
from .filters.kalman_filter import KalmanFilter_2
from .visualization import Visualizer

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
import numpy as np
#from kalman_filter import KalmanFilter_2

class KalmanFilterPureNode(Node):
    def __init__(self):
        super().__init__('kalman_filter_pure_node')

        # TODO: Initialize 6D state and covariance
        initial_state = np.zeros(6)
        initial_covariance = np.eye(6) * 0.1

        self.kf = KalmanFilter_2(initial_state, initial_covariance)

        self.visualizer = Visualizer()
        self.odom_simulator = Odom2DDriftSimulator()
        self.last_time = None

        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/kf2_estimate',
            10
        )

    def odom_callback(self, msg):
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        theta = 2 * np.arctan2(qz, qw)

        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z

        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        #u = np.array([vx, vy, w])
        u = np.array([v, w])

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is None:
            self.last_time = current_time
            return
        dt = current_time - self.last_time
        self.last_time = current_time

        self.kf.predict(u, dt)

        real_pose = odom_to_pose2D(msg)
        noisy_measurement = generate_noisy_measurement_2(real_pose, vx, vy, w)
        
        self.kf.update(noisy_measurement)

        estimated_state = PoseWithCovarianceStamped()
        estimated_state.header.stamp = msg.header.stamp
        estimated_state.header.frame_id = 'map'

        estimated_state.pose.pose.position.x = self.kf.mu[0]
        estimated_state.pose.pose.position.y = self.kf.mu[1]
        estimated_state.pose.pose.position.z = 0.0

        ori = self.kf.mu[2]
        q = self.theta_to_quaternion(ori)
        estimated_state.pose.pose.orientation.x = q.x
        estimated_state.pose.pose.orientation.y = q.y
        estimated_state.pose.pose.orientation.z = q.z
        estimated_state.pose.pose.orientation.w = q.w

        cov = np.zeros(36)
        cov[0] = self.kf.Sigma[0, 0]
        cov[1] = self.kf.Sigma[0, 1]
        cov[5] = self.kf.Sigma[0, 2]
        cov[6] = self.kf.Sigma[1, 0]
        cov[7] = self.kf.Sigma[1, 1]
        cov[35] = self.kf.Sigma[2, 2]
        estimated_state.pose.covariance = cov.tolist()

        self.publisher.publish(estimated_state)

        self.visualizer.update(
            real_pose=real_pose,
            estimated_pose=self.kf.mu[:3],
            covariance=self.kf.Sigma[:3, :3],
            step="update"
        )

    def theta_to_quaternion(self, theta):
        q = Quaternion()
        q.w = math.cos(theta / 2)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(theta / 2)
        return q
    
        

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilterPureNode()
    rclpy.spin(node)
    rclpy.shutdown()

