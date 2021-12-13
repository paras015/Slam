#!/usr/bin/env python3

from fsd_common_msgs.msg import ConeDetections, ConeOdom, Cone
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped
import rospy
import matplotlib.pyplot as plt
import numpy as np
import time


class Particle:
    def __init__(self):
        self.w = 1.0 / 100
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        # landmark x-y positions
        self.lm = np.zeros((0, 2))
        # landmark position covariance
        self.lmP = np.zeros((0 * 2, 2))


class FastSLAM():
    def __init__(self):
        self.Q = np.diag([3.0, np.deg2rad(10.0)]) ** 2
        self.MAX_RANGE = 20.0  # simulation time [s]
        self.M_DIST_TH = 2.0  # maximum observation range
        self.STATE_SIZE = 3  # State size [x,y,yaw]
        self.LM_SIZE = 2  # LM state size [x,y]
        self.N_PARTICLE = 100  # number of particle
        self.NTH = self.N_PARTICLE / 1.5  # Number of particle for re-sampling
        self.velocity = 0
        self.yawrate = 0
        self.time_start = "new"
        self.percepetion_data = []
        self.particles = [Particle() for _ in range(self.N_PARTICLE)]
        self.initializePublisher()

    def initializePublisher(self):
        self.pub = rospy.Publisher(
            '/ekf_slam_output', ConeDetections, queue_size=1)

    def initializeSubscribers(self):
        rospy.Subscriber('fsds/imu', Imu, callback=self.yawrateCallback)
        rospy.Subscriber('/perception/lidar/cone_detections', ConeOdom,
                         buff_size=2**24, queue_size=1, callback=self.slamCallback)
        rospy.Subscriber('fsds/gss', TwistStamped,
                         callback=self.velocityCallback)

    def velocityCallback(self, data):
        x_vel = data.twist.linear.x
        y_vel = data.twist.linear.y
        self.velocity = np.sqrt(x_vel ** 2 + y_vel ** 2)

    def yawrateCallback(self, data):
        self.yawrate = data.angular_velocity.z

    def slamCallback(self, data):
        a = time.time()
        time_ = str(data.position.header.stamp.secs) + "." + str(data.position.header.stamp.nsecs)
        time_ = float(time_)
        if self.time_start == "new":
            self.time_diff = 0
            self.time_start = time_
        else:
            self.time_diff = abs(time_ - self.time_start)
            self.time_start = time_
        # print(self.time_diff)
        self.setConesPositions(data)
        control_input = self.setInput()
        observation = self.calculateObservation(self.perception_data)
        particles = self.runFastSlam(self.particles, control_input, observation)
        xEst = self.calc_final_state(particles)
        print("size", len(particles[0].lm))
        print("Running")
        b = time.time()
        print("time", b - a)
        self.publishLandmarks(xEst, particles)

    def setConesPositions(self, data):
        self.perception_data = []

        for i in range(len(data.cone_detections)):
            point = []
            point.append(data.cone_detections[i].position.x)
            point.append(data.cone_detections[i].position.y)
            self.perception_data.append(point)

        self.perception_data = np.array(self.perception_data)

    def calculateObservation(self, perception_data):
        observation = np.zeros((0, 2))
        for i in range(len(perception_data[:, 0])):
            dx = perception_data[i, 0]
            dy = perception_data[i, 1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            ang = np.arctan2(dy, dx) - np.pi / 2
            angle = self.convertPiToPi(angle=ang)
            if distance <= self.MAX_RANGE:
                zi = np.array([distance, angle])
                observation = np.vstack((observation, zi))
        return observation

    def runFastSlam(self, particles, control_input, observations):
        particles = self.predict_particles(particles, control_input)
        particles = self.update_with_observation(particles, observations)
        particles = self.resampling(particles)
        return particles

    def predict_particles(self, particles, control_input):
        for i in range(self.N_PARTICLE):
            px = np.zeros((self.STATE_SIZE, 1))
            px[0, 0] = particles[i].x
            px[1, 0] = particles[i].y
            px[2, 0] = particles[i].yaw
            px = self.motion_model(px, control_input)
            particles[i].x = px[0, 0]
            particles[i].y = px[1, 0]
            particles[i].yaw = px[2, 0]
        return particles

    def motion_model(self, x, u):
        F = np.array([[1.0, 0, 0],
                    [0, 1.0, 0],
                    [0, 0, 1.0]])
        if u[1][0] != 0:
            divide = (u[0][0] / u[1][0])
            control = np.array([[(-divide * np.sin(x[2, 0])) + (divide * np.sin(x[2, 0] + u[1][0] * self.time_diff))],
                                [(divide * np.cos(x[2, 0])) - (divide * np.cos(x[2, 0] + u[1][0] * self.time_diff))],
                                [(u[1][0] * self.time_diff)]])
        else:
            control = np.array([[u[0][0] * self.time_diff],
                                [u[0][0] * self.time_diff],
                                [0]])
        x = F @ x + control
        x[2, 0] = self.convertPiToPi(x[2, 0])
        return x

    def update_with_observation(self, particles, observations):
        # for iz in range(len(z[0, :])):
        for iz in range(len(observations)):
            # landmark_id = int(z[2, iz])
            # landmark_id = int(observations[iz, 2])
            # count = 0
            for ip in range(self.N_PARTICLE):
                # new landmark
                association_index = self.searchCorrespondingLandmarkID(particles[ip], observations[iz, :])
                # print("AA", association_index)
                # print("size", len(particles[ip].lm))
                # if len(particles[ip].lm) == 8:
                #     count += 1
                # if count == 2:
                #     quit()
                if association_index == len(particles[ip].lm):
                    # print("Append")
                    particles[ip] = self.add_new_landmark(particles[ip], observations[iz, :], self.Q)
                # known landmark
                else:
                    w = self.compute_weight(particles[ip], observations[iz, :], self.Q, association_index)
                    particles[ip].w = w
                    particles[ip] = self.update_landmark(particles[ip], observations[iz, :], self.Q, association_index)
        return particles

    def searchCorrespondingLandmarkID(self, particle, observation):
        number_of_landmarks = len(particle.lm)
        distances = []
        state = np.array([particle.x, particle.y, particle.yaw])
        
        zp = np.array([0, 0])
        zp[0] = (state[0] + observation[0] * np.cos(state[2] + observation[1]))
        zp[1] = (state[1] + observation[0] * np.sin(state[2] + observation[1]))
        # print("ob", zp)
        for i in range(number_of_landmarks):
            landmark = particle.lm[i]
            # print("land", landmark)
            innovation = zp[:2] - landmark
            # print("inv", innovation)
            distances.append(self.calculateDistances(innovation, particle, i))
        distances.append(10.0)
        association_index = distances.index(min(distances))
        # print("Dist", distances)
        
        return association_index

    def calculateDistances(self, innovation, particle, lm_id):
        try: 
            dist = np.matmul(np.matmul((innovation.T), (np.linalg.inv(particle.lmP[2 * lm_id:2 * lm_id + 2, :]))), innovation)
            # print("mat", particle.lmP[2 * lm_id:2 * lm_id + 2, :])
            # print("sidt", dist)
            return dist
        except np.linalg.linalg.LinAlgError:
            # print("singular")
            return 1000000000

    def resampling(self, particles):
        """
        low variance re-sampling
        """

        particles = self.normalize_weight(particles)

        pw = []
        for i in range(self.N_PARTICLE):
            pw.append(particles[i].w[0][0])
        pw = np.array(pw)
        n_eff = 1.0 / (pw @ pw.T)  # Effective particle number
        if n_eff < self.NTH:  # resampling
            w_cum = np.cumsum(pw)
            base = np.cumsum(pw * 0.0 + 1 / self.N_PARTICLE) - 1 / self.N_PARTICLE
            resample_id = base + np.random.rand(base.shape[0]) / self.N_PARTICLE

            inds = []
            ind = 0
            for ip in range(self.N_PARTICLE):
                while (ind < w_cum.shape[0] - 1) \
                        and (resample_id[ip] > w_cum[ind]):
                    ind += 1
                inds.append(ind)

            tmp_particles = particles[:]
            for i in range(len(inds)):
                particles[i].x = tmp_particles[inds[i]].x
                particles[i].y = tmp_particles[inds[i]].y
                particles[i].yaw = tmp_particles[inds[i]].yaw
                particles[i].lm = tmp_particles[inds[i]].lm[:, :]
                particles[i].lmP = tmp_particles[inds[i]].lmP[:, :]
                particles[i].w = 1.0 / self.N_PARTICLE

        return particles

    def normalize_weight(self, particles):
        sum_w = sum([p.w for p in particles])

        try:
            for i in range(self.N_PARTICLE):
                particles[i].w /= sum_w
        except ZeroDivisionError:
            for i in range(self.N_PARTICLE):
                particles[i].w = 1.0 / self.N_PARTICLE

            return particles

        return particles

    def add_new_landmark(self, particle, z, Q_cov):
        r = z[0]
        b = z[1]
        # lm_id = int(z[2])

        s = np.sin(self.convertPiToPi((particle.yaw + b)))
        c = np.cos(self.convertPiToPi((particle.yaw + b)))
        landmark = np.zeros((1, 2))
        landmark[0, 0] = particle.x + r * c
        landmark[0, 1] = particle.y + r * s
        # np.append(particle.lm, landmark, axis=0)
        particle.lm = np.vstack((particle.lm, landmark))
        # print(landmark)
        # print(landmark)
        # particle.lm.append(landmark)
        # particle.lm[lm_id, 0] = particle.x + r * c
        # particle.lm[lm_id, 1] = particle.y + r * s

        # covariance
        dx = r * c
        dy = r * s
        d2 = dx**2 + dy**2
        d = np.sqrt(d2)
        Gz = np.array([[dx / d, dy / d],
                    [-dy / d2, dx / d2]])
        covariance = np.linalg.inv(Gz) @ Q_cov @ np.linalg.inv(Gz.T)
        particle.lmP = np.vstack((particle.lmP, covariance))
        # np.append(particle.lmP, covariance, axis=0)
        return particle

    def compute_weight(self, particle, z, Q_cov, association_index):
        # lm_id = int(z[2])
        xf = np.array(particle.lm[association_index, :]).reshape(2, 1)
        Pf = np.array(particle.lmP[2 * association_index:2 * association_index + 2])
        zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf, Q_cov)

        dx = z[0:2].reshape(2, 1) - zp
        dx[1, 0] = self.convertPiToPi(dx[1, 0])

        try:
            invS = np.linalg.inv(Sf)
        except np.linalg.linalg.LinAlgError:
            print("singular")
            return 1.0

        num = np.exp(-0.5 * dx.T @ invS @ dx)
        # den = 2.0 * np.pi * np.sqrt(np.linalg.det(Sf))
        den = 2.0 * np.pi * np.linalg.det(Sf)
        w = num / den

        return w
    
    def calc_final_state(self, particles):
        xEst = np.zeros((self.STATE_SIZE, 1))

        particles = self.normalize_weight(particles)

        for i in range(self.N_PARTICLE):
            xEst[0, 0] += particles[i].w * particles[i].x
            xEst[1, 0] += particles[i].w * particles[i].y
            xEst[2, 0] += particles[i].w * particles[i].yaw

        xEst[2, 0] = self.convertPiToPi(xEst[2, 0])
        #  print(xEst)

        return xEst

    def update_landmark(self, particle, z, Q_cov, association_index):
        # lm_id = int(z[2])
        xf = np.array(particle.lm[association_index, :]).reshape(2, 1)
        Pf = np.array(particle.lmP[2 * association_index:2 * association_index + 2, :])

        zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf, self.Q)

        dz = z[0:2].reshape(2, 1) - zp
        dz[1, 0] = self.convertPiToPi(dz[1, 0])

        xf, Pf = self.update_kf(xf, Pf, dz, Q_cov, Hf)

        particle.lm[association_index, :] = xf.T
        particle.lmP[2 * association_index:2 * association_index + 2, :] = Pf

        return particle

    def update_kf(self, xf, Pf, v, Q_cov, Hf):
        # PHt = Pf @ Hf.T
        # S = Hf @ PHt + Q_cov

        # S = (S + S.T) * 0.5
        # s_chol = np.linalg.cholesky(S).T
        # s_chol_inv = np.linalg.inv(s_chol)
        # W1 = PHt @ s_chol_inv
        # W = W1 @ s_chol_inv.T

        # x = xf + W @ v
        # P = Pf - W1 @ W1.T

        PHt = Pf @ Hf.T
        S = Hf @ PHt + Q_cov
        K = PHt @ np.linalg.inv(S)
        x = xf + K @ v
        P = (np.eye(2) - K @ Hf) @ Pf

        return x, P

    def compute_jacobians(self, particle, xf, Pf, Q_cov):
        dx = xf[0, 0] - particle.x
        dy = xf[1, 0] - particle.y
        d2 = dx ** 2 + dy ** 2
        d = np.sqrt(d2)
        ang = np.arctan2(dy, dx) - particle.yaw
        zp = np.array(
            [d, self.convertPiToPi(ang)]).reshape(2, 1)

        Hv = np.array([[-dx / d, -dy / d, 0.0],
                    [dy / d2, -dx / d2, -1.0]])

        Hf = np.array([[dx / d, dy / d],
                    [-dy / d2, dx / d2]])

        Sf = Hf @ Pf @ Hf.T + Q_cov

        return zp, Hv, Hf, Sf

    def setInput(self):
        v = self.velocity  # [m/s]
        yawrate = self.yawrate  # [rad/s]
        control_input = np.array([[v, yawrate]]).T
        print(control_input)
        return control_input

    def convertPiToPi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def publishLandmarks(self, xEst, particles):
        cd = ConeDetections()
        cones = []
        for i in range(self.N_PARTICLE):
            for j in range(len(particles[i].lm)):
                co = Cone()
                co.position.x = particles[i].lm[j][0]
                co.position.y = particles[i].lm[j][1]
                co.color.data = "b"
                cones.append(co)
        cd.cone_detections = cones
        cd.header.stamp = rospy.Time.now()
        self.pub.publish(cd)