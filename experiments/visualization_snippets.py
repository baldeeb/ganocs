import matplotlib.pyplot as plt
import numpy as np


def plot_pose_vector(ax, pose, color='r', label=None, scale=1):
    o = np.array([0, 0, 0, 1])
    v = np.array([scale, scale, scale, 1])
    o = pose @ o
    v = o + (pose @ v)
    ov = np.stack([o, v], axis=1)
    ax.label = label
    ax.plot(*ov[:3], color=color)
    # ax.plot(*o[:3], marker='X', color=color)
    ax.plot(*v[:3], marker='*', color=color)


def plot_poses(poses):
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    labels = np.arange(len(poses))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for pose, color, label in zip(poses, colors, labels):
        plot_pose_vector(ax, pose, color=color, label=label)
    plt.show() 