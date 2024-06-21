import numpy as np
import json
import h5py

def split_data(hdf5_path, train_ratio=0.85, test_ratio=0.15):
    with h5py.File(hdf5_path, 'r') as hdf_file:
        # Assuming trajectories or data units are top-level groups in the HDF5 file
        keys = list(hdf_file.keys())
        total_items = len(keys)
        
        # Generate a shuffled array of indices
        indices = np.arange(total_items)
        np.random.shuffle(indices)
        
        # Calculate split sizes
        train_end = int(train_ratio * total_items)
        test_end = train_end + int(test_ratio * total_items)
        
        # Split the indices
        train_indices = indices[:train_end]
        test_indices = indices[train_end:]
        
        # Convert indices back to keys (assuming order in keys list is stable and matches original order)
        train_keys = [keys[i] for i in train_indices]
        test_keys = [keys[i] for i in test_indices]
        
        return train_keys, test_keys

def env_folds(hdf5_path, train_folds, test_fold):
    train_keys = []
    test_keys = []
    with h5py.File(hdf5_path, 'r') as hdf_file:
        # Iterate over each group in the HDF5 file, each group represents a trajectory
        for trajectory_name in hdf_file:
            trajectory_group = hdf_file[trajectory_name]
            
            # Get the first group name from sorted keys
            first_timestep_name = sorted(trajectory_group.keys())[0]
            first_timestep_group = trajectory_group[first_timestep_name]
            
            # Read JSON data from the 'metadata' attribute of the first timestep
            if 'metadata' in first_timestep_group.attrs:
                json_data = json.loads(first_timestep_group.attrs['metadata'])
                if json_data['scene'] in train_folds:
                    train_keys.append(trajectory_name)
                else:
                    test_keys.append(trajectory_name)

    return train_keys, test_keys


def task_gen(hdf5_path, train_ratio=0.85, test_ratio=0.15):
    scene1 = []
    scene2 = []
    scene3 = []
    scene4 = []
    scene5 = []
    with h5py.File(hdf5_path, 'r') as hdf_file:
        # Iterate over each group in the HDF5 file, each group represents a trajectory
        for trajectory_name in hdf_file:
            trajectory_group = hdf_file[trajectory_name]
            
            # Get the first group name from sorted keys
            first_timestep_name = sorted(trajectory_group.keys())[0]
            first_timestep_group = trajectory_group[first_timestep_name]
            
            # Read JSON data from the 'metadata' attribute of the first timestep
            if 'metadata' in first_timestep_group.attrs:
                json_data = json.loads(first_timestep_group.attrs['metadata'])
                if json_data['scene'] == 'FloorPlan_Train1_3':
                    scene1.append(trajectory_name)
                elif json_data['scene'] == 'FloorPlan_Train5_1':
                    scene2.append(trajectory_name)
                elif json_data['scene'] == 'FloorPlan_Train7_5':
                    scene3.append(trajectory_name)
                elif json_data['scene'] == 'FloorPlan_Train8_1':
                    scene4.append(trajectory_name)
                else:
                    scene5.append(trajectory_name)

        np.random.shuffle(np.array(scene1))
        np.random.shuffle(np.array(scene2))
        np.random.shuffle(np.array(scene3))
        np.random.shuffle(np.array(scene4))
        np.random.shuffle(np.array(scene5))

        scene1_end = int(train_ratio * len(scene1))
        scene2_end = int(train_ratio * len(scene2))
        scene3_end = int(train_ratio * len(scene3))
        scene4_end = int(train_ratio * len(scene4))
        scene5_end = int(train_ratio * len(scene5))

        train_keys = scene1[:scene1_end] + scene2[:scene2_end] + scene3[:scene3_end] + scene4[:scene4_end] + scene5[:scene5_end]
        test_keys = scene1[scene1_end:] + scene2[scene2_end:] + scene3[scene3_end:] + scene4[scene4_end:] + scene5[scene5_end:]

    return train_keys, test_keys


def div(hdf5_path, train_runs, test_run):
    test_trajs = []
    scene1 = []
    scene2 = []
    scene3 = []
    scene4 = []
    div_by_two = 52
    div_by_three = 34
    div_by_four = 26
    with h5py.File(hdf5_path, 'r') as hdf_file:
        # Iterate over each group in the HDF5 file, each group represents a trajectory
        for trajectory_name in hdf_file:
            trajectory_group = hdf_file[trajectory_name]
            
            # Get the first group name from sorted keys
            first_timestep_name = sorted(trajectory_group.keys())[0]
            first_timestep_group = trajectory_group[first_timestep_name]
            
            # Read JSON data from the 'metadata' attribute of the first timestep
            if 'metadata' in first_timestep_group.attrs:
                json_data = json.loads(first_timestep_group.attrs['metadata'])
                if json_data['scene'] == test_run:
                    test_trajs.append(trajectory_name)
                elif json_data['scene'] == 'FloorPlan_Train8_1':
                    scene1.append(trajectory_name)
                elif json_data['scene'] == 'FloorPlan_Train1_3':
                    scene2.append(trajectory_name)
                elif json_data['scene'] == 'FloorPlan_Train5_1':
                    scene3.append(trajectory_name)
                else:
                    scene4.append(trajectory_name)
    
    if len(train_runs) == 1:
        train_trajs = np.array(scene1)
    elif len(train_runs) == 2:
        train_trajs = np.concatenate([
            np.random.choice(scene1, size=div_by_two, replace=False),
            np.random.choice(scene2, size=div_by_two, replace=False),
        ])
    elif len(train_runs) == 3:
        train_trajs = np.concatenate([
            np.random.choice(scene1, size=div_by_three, replace=False),
            np.random.choice(scene2, size=div_by_three+1, replace=False),
            np.random.choice(scene3, size=div_by_three+1, replace=False)
        ])
    elif len(train_runs) == 4:
        train_trajs = np.concatenate([
            np.random.choice(scene1, size=div_by_four, replace=False),
            np.random.choice(scene2, size=div_by_four, replace=False),
            np.random.choice(scene3, size=div_by_four, replace=False),
            np.random.choice(scene4, size=div_by_four, replace=False)

        ])

    return train_trajs.tolist(), test_trajs


def remove_spaces(s):
    cs = ' '.join(s.split())
    return cs


def remove_spaces_and_lower(s):
    cs = remove_spaces(s)
    cs = cs.lower()
    return cs
