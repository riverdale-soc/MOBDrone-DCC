'''
Utility functions for the project.
Author: Dmitri Lyalikov
Date: 2023-11-30
'''


from math import *
import numpy as np

DEGREES_PER_METER = 360 / (2 * pi * 6.317e6)

def matrix_rot_x(theta):
    """
    Generate a rotation matrix around the x-axis.

    Parameters:
    - theta (float): The angle of rotation in radians.

    Returns:
    - np.array: The rotation matrix.
    """
    s, c = sin(theta), cos(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]])

def matrix_rot_y(theta):
    """
    Generate a rotation matrix around the y-axis.

    Parameters:
    - theta (float): The angle of rotation in radians.

    Returns:
    - np.array: The rotation matrix.
    """
    s, c = sin(theta), cos(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]])

def matrix_rot_z(theta):
    """
    Generate a rotation matrix around the z-axis.

    Parameters:
    - theta (float): The angle of rotation in radians.

    Returns:
    - np.array: The rotation matrix.
    """
    s, c = sin(theta), cos(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]])

def gstreamer_pipeline(capture_size=(1280, 720), display_size=(1280, 720), framerate=60, flip_method=2):
    """
    Generate a gstreamer pipeline string for video capture.

    Parameters:
    - capture_size (tuple): The size of the captured video (width, height).
    - display_size (tuple): The size of the displayed video (width, height).
    - framerate (int): The framerate of the video.
    - flip_method (int): The flip method for the video.

    Returns:
    - str: The gstreamer pipeline string.
    """
    return (
        'nvarguscamerasrc exposuretimerange="1000000 1000000" maxperf=true ! '
        'video/x-raw(memory:NVMM), '
        'width=(int)%d, height=(int)%d, '
        'format=(string)NV12, framerate=(fraction)%d/1 ! '
        'nvvidconv flip-method=%d ! '
        'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
        'videoconvert ! '
        'video/x-raw, format=(string)BGR ! appsink'
        % (*capture_size, framerate, flip_method, *display_size)
    )

if __name__ == '__main__':
    pt = np.array([0, 0, 1])
    cam_transform = matrix_rot_x(np.radians(-50))
    print(pt @ cam_transform)