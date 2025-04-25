import os
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment

def match_keypoints(kp_start, kp_end):
    """
    Match keypoints between two frames using the Hungarian algorithm.
    """
    num_start = kp_start.shape[0]  
    num_end = kp_end.shape[0]      
    cost_matrix = np.zeros((num_start, num_end))

    for i in range(num_start):
        for j in range(num_end):
            cost_matrix[i, j] = np.linalg.norm(keypoints_start[i] - keypoints_end[j], axis=-1).sum()

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    best_match_idx = np.argmin(cost_matrix)  # Find the index of the minimal cost
    row_best, col_best = divmod(best_match_idx, num_end)

    return row_best, col_best

def spline_interpolate_keypoints(kpts_start, kpts_end, num_frames):
    num_joints = kpts_start.shape[0]
    interpolated_frames = np.zeros((num_frames, num_joints, 3))

    t = np.array([0, 1])
    t_new = np.linspace(0, 1, num_frames+2)[1:-1]

    for joint in range(num_joints):
        for dim in range(3):
            cs = CubicSpline(t, [kpts_start[joint, dim], kpts_end[joint, dim]], bc_type='clamped')
            interpolated_frames[:, joint, dim] = cs(t_new)

    return interpolated_frames


if __name__ == "__main__":
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
    max_gap = 30

    for person_id in range(12, 13):
        print(f"==========\nProcessing Person{str(person_id).zfill(3)}")
        for activity in activities:
            if activity.startswith("_"):
                kitchen_sensor = kitchen_sensors[person_id-1]
                activity = "K" + str(kitchen_sensor) + activity

            print("---------\nActivity: " + activity)

            keypoints_file = os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/pseudocolour_keypoints.json")

            with open(keypoints_file, "r") as f:
                keypoints = json.load(f)
            
            num_kpts_orig = len(keypoints)

            if len(keypoints) == 0:
                print("No keypoints found")
                continue

            while True:
                interpolated_keypoints = dict(keypoints)
                frame_ids = sorted(map(int, keypoints.keys()))
                updated = False

                for i in range(len(frame_ids) - 1):
                    frame_start, frame_end = frame_ids[i], frame_ids[i + 1]
                    keypoints_start = np.array(keypoints[str(frame_start)])
                    keypoints_end = np.array(keypoints[str(frame_end)])
                    frame_diff = frame_end - frame_start

                    if 1 < frame_diff <= max_gap:
                        matched_start_idx, matched_end_idx = match_keypoints(keypoints_start, keypoints_end)
                        keypoints_start = keypoints_start[matched_start_idx:matched_start_idx+1][0]
                        keypoints_end = keypoints_end[matched_end_idx:matched_end_idx+1][0]
                        
                        interpolated_frames = spline_interpolate_keypoints(keypoints_start, keypoints_end, frame_diff)
                        for j in range(1, frame_diff):
                            interpolated_keypoints[str(frame_start + j)] = [interpolated_frames[j].tolist()]
                        updated = True

                if not updated:
                    print(f"Interpolated {len(interpolated_keypoints)} frames from {num_kpts_orig} frames")
                    break

                keypoints = interpolated_keypoints
            
            interpolated_keypoints = {str(k): interpolated_keypoints[str(k)] for k in sorted(map(int, interpolated_keypoints.keys()))}

            with open(os.path.join(base_dir, f"Person{str(person_id).zfill(3)}/{activity}/pseudocolour_keypoints_interp_spline.json"), 'w') as f:
                json.dump(interpolated_keypoints, f, indent=4)

        

