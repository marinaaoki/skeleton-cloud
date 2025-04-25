import cv2
import glob
import os
import json
import numpy as np
from SimpleHRNet import SimpleHRNet
from pathlib import Path
from SimpleHRNet.misc.visualization import draw_points_and_skeleton, joints_dict

model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth", device="cuda:0")

person_id = 13
kitchen_sensors = [1,1,1,2,2,2,1,1,1,1,2,2,1,2,2,1]

activities = [
    "B1_BED_OUT",
    "B2_BED_OUT",
    "B1_JACKET_ON",
    "B2_JACKET_ON",
    "_FRIDGE_OPEN",
    "_FRIDGE_CLOSE",
    "_PREPARE_MEAL",
    "D1_WATER",
    "D2_WATER",
    "D1_EAT",
    "D2_EAT",
    "L1_SIT_DOWN",
    "L2_SIT_DOWN",
    "L1_WATCH_TV",
    "L2_WATCH_TV",
    "L1_STAND_UP",
    "L2_STAND_UP",
    "L1_FALL_DOWN",
    "L2_FALL_DOWN",
    "E1_SHOES_ON",
    "E1_LEAVE_HOUSE",
    "E1_ENTER_HOUSE",
    "E1_SHOES_OFF",
    "W1_BRUSH_TEETH",
    "F1_TAKE_BATH",
    "F1_CLEAN_BATH",
    "B1_JACKET_OFF",
    "B2_JACKET_OFF",
    "B1_BED_IN",
    "B2_BED_IN",
]

base_dir = "/path/to/SITC/"

for person_id in range(12, 13):

    for activity in activities:
        if activity.startswith("_"):
            kitchen_sensor = kitchen_sensors[person_id-1]
            activity = "K" + str(kitchen_sensor) + activity

        print("==========\nProcessing", activity)

        image_files = glob.glob(os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/depthImages/*.png"))
        image_files = sorted(image_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        json_file = os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/pseudocolour_keypoints.json")


        keypoints = {}
        for image_file in image_files:
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            orig_img = image.copy()
            image = cv2.resize(image, (384, 288))
            image = cv2.equalizeHist(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.convertScaleAbs(image, alpha=255/image.max())
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

            joints = model.predict(image)
            joints = np.array(joints)

            if len(joints) == 0:
                continue

            frame_id = int(image_file.split("/")[-1].split(".")[0])
            resized_image = cv2.resize(image, (orig_img.shape[1], orig_img.shape[0]))
            #if not os.path.exists(os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/preprocessed_depthImages/")):
            #    os.makedirs(os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/preprocessed_depthImages/"))
            #cv2.imwrite(os.path.join(base_dir,f"Person{str(person_id).zfill(3)}/{activity}/preprocessed_depthImages/"+image_file.split("/")[-1]), resized_image)

            joints[:, :, 0] = joints[:, :, 0] * orig_img.shape[0] / image.shape[0]
            joints[:, :, 1] = joints[:, :, 1] * orig_img.shape[1] / image.shape[1]

            keypoints[frame_id] = joints.tolist()
        
        print("============")

        if not os.path.exists(os.path.dirname(json_file)):
            os.makedirs(os.path.dirname(json_file))
            
        with open(json_file, 'w') as f:
            json.dump(keypoints, f)
            print("Saved keypoints for " + str(len(image_files)) + " frames to " + json_file)

    