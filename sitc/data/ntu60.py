from pathlib import Path
import numpy as np
import open3d as o3d
import scipy
import torch
import re
import os

class NTU60(torch.utils.data.Dataset):
    """
    Dataset for loading skeleton samples from the NTU60 dataset.
    """
    dataset_path = Path("sitc/data")

    def __init__(self, num_frames=75, num_joints=25,split="train", paired=False):
        """
        Args:
            num_frames (int): Number of frames to sample from each sequence. If the sequence has fewer frames, the last frame will be repeated until the desired number of frames is reached.
            split (str): Dataset split to use. Can be "train" or "val".
            paired (bool): If True, the dataset will return paired samples for each item. If False, the dataset will return single samples.
        """
        super().__init__()
        assert split in ["train", "val"]

        self.max_frames = num_frames
        self.num_joints = num_joints
        self.paired = paired

        self.items = Path(os.path.join(NTU60.dataset_path, f"splits/ntu60_{split}.txt")).read_text().splitlines()
        
        # Define the adjacency matrix for the skeleton joints
        j_pair_1 = np.array([4, 3, 3, 21, 21, 5, 5, 6, 6, 7, 7, 8, 8, 22, 8, 23, 21, 9, 9, 10, 10, 11, 11, 
                     12, 12, 24, 12, 25, 21, 2, 2, 1, 1, 13, 13, 14, 14, 15, 15, 16, 1, 17, 17, 18, 
                     18, 19, 19, 20])
        j_pair_2 = np.array([3, 4, 21, 3, 5, 21, 6, 5, 7, 6, 8, 7, 22, 8, 23, 8, 9, 21, 10, 9, 11, 10, 12, 
                     11, 24 ,12, 25, 12, 2, 21, 1, 2, 13, 1, 14, 13, 15, 14, 16, 15, 17, 1, 18, 17, 
                     19, 18, 20, 19])

        # Subtract 1 from each value to make it 0-indexed.
        j_pair_1 -= 1
        j_pair_2 -= 1

        con_matrix = np.ones([len(j_pair_1)])

        self.adj_joint = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(25, 25)).toarray()

    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the item to retrieve.
        Returns:
            dict: Dictionary containing the following keys:
                - "name" (str): Sample ID.
                - "indices" (int): Index of the item.
                - "keypoints" (dict): Dictionary containing the skeleton data for each frame. The keys are the frame numbers and the values are the skeleton data. The skeleton data is a dictionary where the keys are the subject IDs and the values are the 3D coordinates of the joints.
                - "camera" (int): Integer label of the camera ID.
                - "action" (int): Integer label of the action class.
                - "person" (int): Integer label of the person ID.
        """
        name = self.items[index]
        cam, per, act = re.findall('[A-Z][^A-Z]*', name)[1], re.findall('[A-Z][^A-Z]*', name)[2], re.findall('[A-Z][^A-Z]*', name)[4]
        camera = int(cam.split("C")[1])
        person = int(per.split("P")[1])
        action = int(act.split("A")[1])

        # Load the skeleton data for the given sample
        keypoints_path = f"/media/ubi-lab-desktop/Extreme Pro/data/nturgb+d_skeletons/{name}.skeleton"
        keypoint_samples = self.get_keypoint_samples(keypoints_path)

        if self.paired:
            acs = Path(os.path.join(NTU60.dataset_path, f"NTU_RGBD_{cam}_{act}.txt")).read_text().splitlines()
            acs.remove(name)

            # Choose a random sample from aps
            paired_sample = np.random.choice(acs)

            return {
                "name": name,
                "indices": index,
                "keypoints": keypoint_samples,
                "camera": camera,
                "action": action,
                "person": person,
                "paired": paired_sample
            }


        return {
            "name": name,
            "indices": index,
            "keypoints": keypoint_samples,
            "camera": camera,
            "action": action,
            "person": person
        }


    def __len__(self):
        return len(self.items)
    
    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility function to move a batch to a specified device.
        """
        batch["indices"] = batch["indices"].to(device)
        batch["keypoints"] = batch["keypoints"].to(device)
        batch["camera"] = batch["camera"].to(device)
        batch["action"] = batch["action"].to(device)
        batch["person"] = batch["person"].to(device)

        return batch


    def get_keypoint_samples(self, keypoints_path):
        """
        Load the skeleton data from a given file.
        """
        with open(keypoints_path, "r") as f:
            read_lines = [line.split() for line in f.readlines()]
            num_frames = int(read_lines[0][0])

        keypoint_samples = []

        idx = 0
        for frame in range(1, num_frames+1): # +1 since the loop iterates only till num_of_frames - 1
            idx += 1 #next line 
            num_subjects = int(read_lines[idx][0])
            
            temp_sub = {}
            
            for sub in range(num_subjects):
                idx += 1 #next line 
                sub_id = read_lines[idx][0]
                idx += 1 #next line 

                
                temp_skel = []
                for skeleton in range(int(read_lines[idx][0])):
                    
                    idx += 1 #next line 
                    jointx = float(read_lines[idx][0]) 
                    jointy = float(read_lines[idx][1]) 
                    jointz = float(read_lines[idx][2]) 
                    
                    temp_skel.append((jointx, jointy, jointz))

                # only add the first skeleton
                if sub == 0:
                    keypoint_samples.append(np.asarray(temp_skel)) #store the list as np array.

        # Sample the desired number of frames
        n_frames = len(keypoint_samples)

        if n_frames < self.max_frames:
            # Repeat the last frame until the desired number of frames is reached
            last_frame = keypoint_samples[n_frames - 1]
            for i in range(n_frames, self.max_frames):
                keypoint_samples.append(last_frame)
            n_frames = self.max_frames

        elif n_frames > self.max_frames:
            # Cut the data once the desired number of frames is reached
            keypoint_samples = [keypoint_samples[i] for i in range(self.max_frames)]

        # Swap T and J, so that the shape is (J, T, 3)
        #keypoint_samples = np.swapaxes(np.array(keypoint_samples), 0, 1)

        return np.array(keypoint_samples)