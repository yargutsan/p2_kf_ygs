import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from irobot_create_msgs.msg import WheelVels

import numpy as np
import math

from .sensor_utils import odom_to_pose2D, get_normalized_pose2D, Odom2DDriftSimulator
from .visualization import Visualizer
from .filters.kalman_filter import KalmanFilter 

class KalmanFilterNode(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')

        # TODO: Initialize filter with initial state and covariance
        initial_state = np.zeros(3)
        initial_covariance = np.eye(3) * 0.1

        self.kf = KalmanFilter(initial_state, initial_covariance)

        self.visualizer = Visualizer()
        self.odom_simulator = Odom2DDriftSimulator()

        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/kf_estimate',
            10
        )

    def odom_callback(self, msg):
        # Extraer velocidades lineales y angulares
        linear = msg.twist.twist.linear
        angular = msg.twist.twist.angular.z
        u = np.array([linear.x, angular])  # [vx, omega]

        # Calcular el intervalo de tiempo (delta_t)
        current_time = msg.header.stamp
        if not hasattr(self, 'last_time') or self.last_time is None:
            self.last_time = current_time
            return
        delta_t = (current_time.sec - self.last_time.sec) + \
                (current_time.nanosec - self.last_time.nanosec) * 1e-9
        self.last_time = current_time

        # Paso de predicción del filtro de Kalman
        self.kf.predict(u, delta_t)

        # Obtener la pose 2D directamente desde el mensaje Odometry
        pose_2d = odom_to_pose2D(msg)  # Se asume que devuelve un objeto con .x, .y, .theta

        # Generar una medición ruidosa
        current_time_float = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        noisy_pose = self.odom_simulator.add_drift(pose_2d, current_time_float)
    
        self.kf.update(noisy_pose)
        self.visualizer.update(pose_2d, self.kf.mu, self.kf.Sigma, step="update")

        # Publicar el estado estimado como PoseWithCovarianceStamped
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = msg.header.stamp
        pose_msg.header.frame_id = 'map'

        # Posición estimada
        pose_msg.pose.pose.position.x = self.kf.mu[0]
        pose_msg.pose.pose.position.y = self.kf.mu[1]
        pose_msg.pose.pose.position.z = 0.0

        # Convertir ángulo theta a cuaternión (2D: solo componente z y w)
        theta = self.kf.mu[2]
        pose_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        pose_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

        # Matriz de covarianza 
        covariance_6x6 = [0.0] * 36

        # Asignar valores de la matriz de covarianza estimada (3x3) a la 6x6
        # Map: x → [0,1,5], y → [6,7,11], theta (yaw) → [30,31,35]
        covariance_6x6[0]  = self.kf.Sigma[0, 0]  # x-x
        covariance_6x6[1]  = self.kf.Sigma[0, 1]  # x-y
        covariance_6x6[5]  = self.kf.Sigma[0, 2]  # x-theta

        covariance_6x6[6]  = self.kf.Sigma[1, 0]  # y-x
        covariance_6x6[7]  = self.kf.Sigma[1, 1]  # y-y
        covariance_6x6[11] = self.kf.Sigma[1, 2]  # y-theta

        covariance_6x6[30] = self.kf.Sigma[2, 0]  # theta-x
        covariance_6x6[31] = self.kf.Sigma[2, 1]  # theta-y
        covariance_6x6[35] = self.kf.Sigma[2, 2]  # theta-theta

        pose_msg.pose.covariance = covariance_6x6


        # Publicar el mensaje estimado
        self.publisher.publish(pose_msg)

        


def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilterNode()
    rclpy.spin(node)
    rclpy.shutdown()
