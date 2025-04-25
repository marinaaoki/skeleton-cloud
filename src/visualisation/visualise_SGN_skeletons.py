import numpy as np
import cv2
import h5py
import pickle
from misc.visualization import draw_points_and_skeleton, joints_dict

ntu_cv = "/path/to/SGN/data/sitc/SITC_TC2-c.h5"
f = h5py.File(ntu_cv, 'r')
X = f['test']['x'][:]
#X = f['x'][:]
sample = X[0]
print(sample.shape)

image_size = 512

for i, sample_frame in enumerate(sample):
    if sample_frame.sum() == 0:
        break
    try:
        joints = np.array(joints)
        joints = joints + 256
        joints = joints - joints.min()
        joints = joints / joints.max()
        joints = (joints * image_size).astype(int)
    except:
        joints = None

    if joints is not None:
        image = np.zeros((image_size, image_size, 3), np.uint8)
        image = draw_points_and_skeleton(image, joints, joints_dict()["ntu60"]['skeleton'], points_color_palette='spring', skeleton_color_palette='Pastel1')
        alpha_image = np.zeros((image_size, image_size, 4), np.uint8)
        alpha_image[:, :, 3] = np.where(image[:, :, 0] == 0, 0, 255)
        alpha_image[:, :, 0:3] = image
        # rotate image 90 degrees counter-clockwise
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imshow("image", image)
    
        cv2.waitKey(0)

cv2.destroyAllWindows()
