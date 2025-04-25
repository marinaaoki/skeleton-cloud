import os
import json

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

activites = ["E1_ENTER_HOUSE"]
base_dir = "/path/to/SITC"


# in this script, we will go through each activity folder and re-index the keypoints stored in the pseudocolour_keypoints.json file according to the cut_indices.txt
# if the beginning of the activity was cut, we need to subtract the frame id by the cut index
# if the end of the activity was cut, we need to remove subsequent frames from the keypoints
# finally, we re-index the frame ids in the depthImages directory to match the new frame ids in the keypoints file

for person_id in range(4, 5):
    print(f"==========\nProcessing Person{str(person_id).zfill(3)}")
    for activity in activities:
        if activity.startswith("_"):
            kitchen_sensor = kitchen_sensors[person_id-1]
            activity = "K" + str(kitchen_sensor) + activity

        if not os.path.exists(os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/cut_indices.txt")):
            continue

        print("---------\nActivity: " + activity)

        with open(os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/cut_indices.txt"), 'r') as f:
            cut_indices = f.readlines()

        with open(os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/pseudocolour_keypoints.json"), 'r') as f:
            keypoints = json.load(f)
            keypoints_new = {}

        print(f"Keypoints: {keypoints.keys()}")

        image_dir = os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/depthImages/")

        images = os.listdir(image_dir)
        images = sorted(images, key=lambda x: int(x.split(".")[0]))

        for cut_idx in cut_indices:
            # if the end of the activity was cut, we remove subsequent frames from the keypoints
            if cut_idx.startswith("end"):
                cut_index = int(cut_idx.split(":")[1])
                print(f"End: {cut_index}")

                for frame_id in keypoints.keys():
                    if int(frame_id) <= cut_index:
                        keypoints_new[frame_id] = keypoints[frame_id]
                    

                print(f"New keypoints: {keypoints_new.keys()}")
                with open(os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/pseudocolour_keypoints_new.json"), 'w') as f:
                    json.dump(keypoints_new, f)

            # if the beginning of the activity was cut, we shift the frame ids for both the keypoints and the images by the cut index
            elif cut_idx.startswith("begin"):
                cut_index = int(cut_idx.split(":")[1])
                print(f"Begin: {cut_index}")
                for frame_id in keypoints.keys():
                    if int(frame_id) > cut_index:
                        keypoints_new[str(int(frame_id)-cut_index)] = keypoints[frame_id]
                # save the new keypoints
                print(f"New keypoints: {keypoints_new.keys()}")
                with open(os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/pseudocolour_keypoints_new.json"), 'w') as f:
                    json.dump(keypoints_new, f)


                #for image in images:
                    #if int(image.split(".")[0]) <= cut_index:
                    #    continue
                    #frame_id = image.split(".")[0]
                    #os.rename(image_dir + image, image_dir + str(int(frame_id)-cut_index) + ".png")
            
            



                