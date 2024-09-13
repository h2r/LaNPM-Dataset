import numpy as np
import json
import h5py
import os
from random import sample, seed
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
from collections import defaultdict
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

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
    np.random.seed(420)
    test_trajs = []
    scene1 = []
    scene2 = []
    scene3 = []
    scene4 = []
    div_by_two = 52
    div_by_three = 31 #26 #35 #39
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
                elif json_data['scene'] == 'FloorPlan_Train12_3':
                    scene2.append(trajectory_name)
                elif json_data['scene'] == 'FloorPlan_Train5_1':
                    scene3.append(trajectory_name)
                else:
                    scene4.append(trajectory_name)
    
    if len(train_runs) == 1:
        train_trajs = np.array(scene4)
    elif len(train_runs) == 2:
        train_trajs = np.concatenate([
            np.random.choice(scene1, size=div_by_two, replace=False),
            np.random.choice(scene2, size=div_by_two, replace=False),
        ])
    elif len(train_runs) == 3:
        train_trajs = np.concatenate([
            np.random.choice(scene1, size=div_by_three, replace=False),
            np.random.choice(scene2, size=div_by_three, replace=False),
            np.random.choice(scene3, size=div_by_three, replace=False)
        ])
    elif len(train_runs) == 4:
        train_trajs = np.concatenate([
            np.random.choice(scene1, size=div_by_four, replace=False),
            np.random.choice(scene2, size=div_by_four, replace=False),
            np.random.choice(scene3, size=div_by_four, replace=False),
            np.random.choice(scene4, size=div_by_four, replace=False)

        ])
    
    test_trajs = np.random.choice(test_trajs, size=22, replace=False)
    return train_trajs.tolist(), test_trajs.tolist()

def cluster(hdf5_path, low_div=True):
    seed(3)

    model = SentenceTransformer("all-mpnet-base-v2") # all-MiniLM-L6-v2, all-MiniLM-L12-v2, all-distilroberta-v1, sentence-t5-base, all-mpnet-base-v2

    if not os.path.exists("sim_commands.npy") or not os.path.exists("command_key_dict.json"):
        commands=[]
        command_key_dict = {}
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf_file:
            # Iterate through each trajectory group
            for key,(trajectory_name, trajectory_group) in zip(hdf_file.keys() ,hdf_file.items()):
                # Iterate through each timestep group within the trajectory
                for timestep_name, timestep_group in trajectory_group.items():
                    # Read and decode the JSON metadata
                    metadata = json.loads(timestep_group.attrs['metadata'])
                    commands.append(metadata['nl_command'])
                    command_key_dict[metadata['nl_command']] = key
                    break

        np.save("sim_commands.npy", commands)
        with open('command_key_dict.json', 'w') as f:
            json.dump(command_key_dict, f)
    else:
        commands = np.load('sim_commands.npy', allow_pickle=True).tolist()
        with open('command_key_dict.json', 'r') as file:
            command_key_dict = json.load(file)

    # Convert the commands to embeddings
    embeddings = model.encode(commands)

    # Compute the cosine similarity matrix
    # sim_mat = cosine_similarity(embeddings)
    # print(sim_mat)

    if not os.path.exists("cluster_dict.json"):
        # Apply Agglomerative Clustering
        clustering = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.3)
        clusters = clustering.fit_predict(embeddings)

        # Calculate the silhouette score
        silhouette_avg = silhouette_score(embeddings, clusters, metric='cosine')
        print(f'Silhouette Score: {silhouette_avg}')

        # Calculate the Davies-Bouldin index
        db_index = davies_bouldin_score(embeddings, clusters)
        print(f'Davies-Bouldin Index: {db_index}')

        # Create a defaultdict with lists as default values
        cluster_dict = defaultdict(list)

        # Populate the dictionary
        for string, cluster_id in zip(commands, clusters):
            cluster_dict[int(cluster_id)].append(string)

        cluster_dict = dict(cluster_dict)

        #save dict
        with open('cluster_dict.json', 'w') as f:
            json.dump(cluster_dict, f)
    else:
        with open('cluster_dict.json', 'r') as file:
            cluster_dict = json.load(file)
    # Find the cluster with the longest list
    sorted_clusters = sorted(cluster_dict, key=lambda k: len(cluster_dict[k]), reverse=True)

    if low_div:
        start = 0
        end = 10
    else: # high div
        start = 14
        end = 106

    # tot = 0
    train_keys = []
    if low_div:
        for i in range(start,end):
            # print(f"{len(cluster_dict[sorted_clusters[i]])} elements.")
            command_lst = cluster_dict[sorted_clusters[i]]
            for cmd in command_lst:
                key = command_key_dict[cmd]
                train_keys.append(key)
            # tot += len(cluster_dict[sorted_clusters[i]])
        #print(f'Total: {tot}')
        num_elements_to_pick = len(train_keys) - 2
        train_keys = sample(train_keys, num_elements_to_pick)
    else:
        for i in range(start,end):
            # print(f"{len(cluster_dict[sorted_clusters[i]])} elements.")
            command_lst = cluster_dict[sorted_clusters[i]]
            for cmd in command_lst:
                key = command_key_dict[cmd]
                train_keys.append(key)
            # tot += len(cluster_dict[sorted_clusters[i]])
        #print(f'Total: {tot}')
        # num_elements_to_pick = len(train_keys) - 
        # train_keys = sample(train_keys, num_elements_to_pick)
    test_keys = []
    for i in range(10,14): #test
        command_lst = cluster_dict[sorted_clusters[i]]
        for cmd in command_lst:
            key = command_key_dict[cmd]
            test_keys.append(key)
    num_elements_to_pick = len(test_keys) - 5
    test_keys = sample(test_keys, num_elements_to_pick)
    return train_keys, test_keys

def remove_spaces(s):
    cs = ' '.join(s.split())
    return cs

def remove_spaces_and_lower(s):
    cs = remove_spaces(s)
    cs = cs.lower()
    return cs