import os
import json
import numpy as np
import h5py
import zipfile
from tqdm import tqdm

# Directory containing the zip files
zip_files_directory = '/users/ajaafar/scratch/lanmp_dataset/Trajectories'
# HDF5 file to create
hdf5_file_path = '/users/ajaafar/scratch/lanmp_dataset/lanmp_dataset.hdf5'

# Function to safely create a group if it doesn't exist
def safe_create_group(hdf_file, group_name):
    if group_name not in hdf_file:
        return hdf_file.create_group(group_name)
    return hdf_file[group_name]

# Safely store metadata in a group if the attribute does not already exist.
def safe_store_metadata(group, attribute_name, data):
    if attribute_name not in group.attrs:
        group.attrs[attribute_name] = data


# Open or create the HDF5 file
with h5py.File(hdf5_file_path, 'a') as hdf_file:
    # Get a list of zip files
    zip_files = sorted([f for f in os.listdir(zip_files_directory) if f.endswith('.zip')])
    for zip_file_index, zip_file_name in enumerate(tqdm(zip_files, desc='Processing Zip Files')):
        trajectory_group_name = os.path.splitext(zip_file_name)[0]  # Use zip file name as group name
        # Check if the trajectory group already exists, skip if it does
        if trajectory_group_name in hdf_file:
            print(f"Skipping already processed {trajectory_group_name}")
            continue
        else:
            trajectory_group = hdf_file.create_group(trajectory_group_name)
        
        # Process the zip file
        zip_file_path = os.path.join(zip_files_directory, zip_file_name)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract the zip file to a temporary directory
            zip_ref.extractall('temp_extract_dir')
            # List directories and sort them to ensure order
            folders = sorted(os.listdir('temp_extract_dir'))
            # Wrap the folder processing with tqdm for a progress bar
            for folder_name in tqdm(folders, desc=f'Folders in {zip_file_name}'):
                timestep_group_name = folder_name
                # Create a group for the timestep
                timestep_group = safe_create_group(trajectory_group, timestep_group_name)

                folder_path = os.path.join('temp_extract_dir', folder_name)
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    # Handle the JSON file
                    if file_name.endswith('.json'):
                        with open(file_path, 'r') as json_file:
                            json_data = json.load(json_file)
                            # Convert the JSON data to a string since HDF5 attributes are better stored as strings
                            json_str = json.dumps(json_data)
                            # Safely store the JSON string as metadata if it hasn't been stored already
                            safe_store_metadata(timestep_group, 'metadata', json_str)
                    # Handle NPY files
                    elif file_name.endswith('.npy'):
                        data = np.load(file_path)
                        dataset_name = os.path.splitext(file_name)[0]
                        if dataset_name not in timestep_group:  # Check if the dataset already exists
                            timestep_group.create_dataset(dataset_name, data=data, compression="gzip")
                
            # After processing a zip file:
            if (zip_file_index + 1) % 10 == 0:
                hdf_file.flush()

            # Cleanup: remove the temporary extracted directory and its contents
            for root, dirs, files in os.walk('temp_extract_dir', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir('temp_extract_dir')