import numpy as np
import cv2
import os
import json
import glob
from misc.visualization import draw_points_and_skeleton, joints_dict

person_id = 13
activity = "K1_FRIDGE_CLOSE"

keypoints_file = f"/u/home/aoma/gr/datasets/SITC/Person{str(person_id).zfill(3)}/{activity}/pseudocolour_keypoints_interp_spline_corr.json"
image_folder = f"/u/home/aoma/gr/datasets/SITC/Person{str(person_id).zfill(3)}/{activity}/depthImages/"

if os.path.exists(keypoints_file):
    with open(keypoints_file, 'r') as f:
        keypoints = json.load(f)  
else:
    with open(f"/u/home/aoma/gr/datasets/SITC/Person{str(person_id).zfill(3)}/{activity}/pseudocolour_keypoints.json") as f:
        keypoints = json.load(f)

image_files = sorted(glob.glob(image_folder + "*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0]))

for image_file in image_files:
    frame_id = int(image_file.split("/")[-1].split(".")[0])

    try:
        joints = keypoints[str(frame_id)]
    except KeyError:
        joints = None

    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    # overlay frame id on the image
    cv2.putText(image, str(frame_id), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

    if joints is not None:
        joints = np.array(joints)
        if joints.shape[0] > 1:
            print(f"frame_id: {frame_id}, joints.shape: {joints.shape}")
        for joints_set in joints:
            image = draw_points_and_skeleton(image, joints_set, joints_dict()["coco"]['skeleton'], points_color_palette='managua_r', skeleton_color_palette='managua',
                                               points_palette_samples=1)
        if not os.path.exists(f"/u/home/aoma/gr/datasets/SITC/Person{str(person_id).zfill(3)}/{activity}/skeletonVisualisation_pseudocolour_spline_corr/"):
            os.makedirs(f"/u/home/aoma/gr/datasets/SITC/Person{str(person_id).zfill(3)}/{activity}/skeletonVisualisation_pseudocolour_spline_corr/")
        # save the image
        cv2.imwrite(f"/u/home/aoma/gr/datasets/SITC/Person{str(person_id).zfill(3)}/{activity}/skeletonVisualisation_pseudocolour_spline_corr/"+f"{frame_id}.png", image)
    
        #cv2.imshow("image", image)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        #cv2.waitKey(0)

cv2.destroyAllWindows()
