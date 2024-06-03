import numpy as np
import json
import h5py

def split_data(hdf5_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    with h5py.File(hdf5_path, 'r') as hdf_file:
        # Assuming trajectories or data units are top-level groups in the HDF5 file
        keys = list(hdf_file.keys())
        total_items = len(keys)
        
        # Generate a shuffled array of indices
        indices = np.arange(total_items)
        np.random.shuffle(indices)
        
        # Calculate split sizes
        train_end = int(train_ratio * total_items)
        val_end = train_end + int(val_ratio * total_items)
        
        # Split the indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Convert indices back to keys (assuming order in keys list is stable and matches original order)
        train_keys = [keys[i] for i in train_indices]
        val_keys = [keys[i] for i in val_indices]
        test_keys = [keys[i] for i in test_indices]
        
        return train_keys, val_keys, test_keys

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