import tf
import math
import numpy as np
from matplotlib import pyplot as plt
alpha, beta, gamma = 0.123, -1.234, 2.345
origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
I = tf.transformations.identity_matrix()
Rx = tf.transformations.rotation_matrix(alpha, xaxis)
Ry = tf.transformations.rotation_matrix(beta, yaxis)
Rz = tf.transformations.rotation_matrix(gamma, zaxis)
R = tf.transformations.concatenate_matrices(Rz, Ry, Rx)
euler = tf.transformations.euler_from_matrix(R, 'sxyz') 
print(euler)

q = tf.transformations.quaternion_from_euler(-0.064668599104332, 0.035643164014983, 1.592362078873741,'sxyz') 
R2 = tf.transformations.quaternion_matrix(q)
print(q)
euler2 = tf.transformations.euler_from_matrix(R2, 'sxyz')
print(euler2)

# help(tf.transformations.quaternion_from_euler)
# help(tf.transformations.rotation_matrix)
# help(tf.transformations)
# print(R)
gt_filename  = "/media/binpeng/BIGLUCK/Datasets/NCLT/datasets/groundtruth_2012-01-08.csv"
pose_gt = np.loadtxt(gt_filename, delimiter = ",")
# # Note: Interpolation is not needed, this is done as a convience
# interp = scipy.interpolate.interp1d(gt[:, 0], gt[:, 1:], kind='nearest', axis=0)
# pose_gt = interp(t_cov)

# NED (North, East Down)
t_pose = pose_gt[:,0]
x = pose_gt[:, 1]
y = pose_gt[:, 2]
z = pose_gt[:, 3]

r = pose_gt[:, 4]
p = pose_gt[:, 5]
h = pose_gt[:, 6]

q_arr = np.zeros((len(t_pose),4))
for i, utime in enumerate(t_pose):
    q = tf.transformations.quaternion_from_euler(r[i],p[i],h[i],'sxyz') 
    q_arr[i,:] = q
plt.plot(t_pose,q_arr[:,2])
plt.show()