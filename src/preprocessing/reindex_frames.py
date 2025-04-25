import os

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

base_dir = "/path/to/SITC"

for person_id in range(11, 12):
    for activity in activities:
        in_order = True
        if activity.startswith("_"):
            kitchen_sensor = kitchen_sensors[person_id-1]
            activity = "K" + str(kitchen_sensor) + activity


        image_dir = os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/depthImages/")


        mages = os.listdir(image_dir)
        images = sorted(images, key=lambda x: int(x.split(".")[0]))

        # Check if the images need to be renamed (i.e. if they are not in ascending order from 0)
        #for i, image in enumerate(images):
        #    if i != int(image.split(".")[0]):
        #        in_order = False
                
        #if not in_order:
        #    print(f"Images are not in ascending order for activity {activity}, person {person_id}")

        for i, image in enumerate(images):
            os.rename(image_dir + image, image_dir + str(i) + ".png")
            
            
