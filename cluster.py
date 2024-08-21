from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics import davies_bouldin_score
import h5py
from sklearn.metrics import silhouette_score

# Load a pre-trained Sentence-BERT model

# model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-MiniLM-L12-v2')
model = SentenceTransformer("all-mpnet-base-v2")
# model = SentenceTransformer('all-distilroberta-v1')
# model = SentenceTransformer('sentence-t5-base')

# Example list of commands
# commands = [
#     "go to the bathroom, get toilet paper, and put it on the couch in the living room",
#     # "go to the bathroom, get toilet paper, and put it on the desk in the living room",
#     # "go to the bathroom, get the phone, take it to the garden, place it on the blue couch"
#     # "take the ball from the living room to the toilet"
#     "library"
#     # Add more commands here
# ]


# path = "/users/ajaafar/data/shared/lanmp/real_dataset.hdf5" #HDF5 dataset path
# commands = []
# # Open the HDF5 file
# with h5py.File(path, 'r') as hdf5:
#     # Print the root groups which are 'LabTrajectories' and 'FloorTrajectories'
#     for trajectory_type in hdf5.keys():
#         # print(f"Trajectory type: {trajectory_type}")

#         # Access each trajectory group within the type
#         for group_name in hdf5[trajectory_type].keys():
#             # print(f"  Group: {group_name}")

#             # Access each sub-group/dataset within the group
#             # i=0
#             for sub_group_name in hdf5[trajectory_type][group_name].keys():
#                 # print(f"    Sub-group/Dataset: {sub_group_name}")
#                 sub_group = hdf5[trajectory_type][group_name][sub_group_name]

#                 # Read and decode the JSON metadata
#                 metadata_json = sub_group['metadata'][()]
#                 metadata = json.loads(metadata_json.decode('utf-8'))
#                 commands.append(metadata['language_command'])
#                 break
#                 # print(f"Metadata: {json.dumps(metadata, indent=2)}")]
#                 # if i ==0:
#                 #     print(metadata['language_command'])
#                 # i+=1


# # HDF5 file to read
# path = "/users/ajaafar/data/shared/lanmp/sim_dataset.hdf5" #HDF5 dataset path
# commands=[]
# # Open the HDF5 file
# with h5py.File(path, 'r') as hdf_file:
#     # Iterate through each trajectory group
#     for trajectory_name, trajectory_group in hdf_file.items():
#         # print(f"Trajectory: {trajectory_name}")
#         # Iterate through each timestep group within the trajectory
#         for timestep_name, timestep_group in trajectory_group.items():
#             # print(f"Step: {timestep_name}")

#             # Read and decode the JSON metadata
#             metadata = json.loads(timestep_group.attrs['metadata'])
#             # print(f"Metadata: {json.dumps(metadata, indent=2)}")
#             # print(metadata['nl_command'])
#             commands.append(metadata['nl_command'])
#             break

# np.save("sim_commands.npy", commands)

# commands = [
#     "Take the cup from the table in the dining area which is closest to the stairs and bring it to the table at the coaches in the corner of the big dining room with besides the windows",
#     "Bring the cleaning bottle for the whiteboard in the sideroom with the tables and the projector to the big hallway which is located behind the kitchen and place it infront of room 311 on the table.",
#     "Take the waterbottle and put it below the stairs and take the soap bottle and put it besides the plant on the right side in the corner with the windows of the dining room",
#     "Take the plates in the kitchen and throw them in the newspaper garbage in the big hallway behind the kitchen which is placed closest to the stairs",
#     "Take the dishsoap in the kitchen and hide it as far under the stairs as you can",
#     "Go pick up Hershey's syrup in the room with the big window and bring it to the room with the other Spot.",
#     "Go get a snack in the room with the fridge and bring it to the room with the big whiteboard." ,
#     "Go pick up a stuffed animal in the room with the big TV and leave it on a chair in the room with the big arms.",
#     "Go pick up the Dasani water bottle in the room with the Spot charging port and drop it off in sink in the room where the fridge is.",
#     "Go pick up the blue book in research lab and leave it on one of the tables in the main room.",
#     "Get the marker from the main room where the couch is and bring it to this room where the other robot is located and place it on the whiteboard.",
#     "Pick up the plastic cups next to the sink in the main room and place them next to the mugs in the room with the claws.",
#     "Get the water bottle next to the monitors from this room and take it to the main room and place it next to the other water bottle next to the doors.",
#     "Get the blue cup from this room next to the monitors and take it to the main room and drop it into the sink.",
#     "Pick up the blue and green shoe from the desk in the main room next to the sink and leave it on the couch in the same room located in the middle of the room."
#     ]

commands = np.load('sim_commands.npy', allow_pickle=True).tolist()
# Convert the commands to embeddings
embeddings = model.encode(commands)

# Compute the cosine similarity matrix
# sim_mat = cosine_similarity(embeddings)
# print(sim_mat)

# Apply Agglomerative Clustering
clustering = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.18)
clusters = clustering.fit_predict(embeddings)

# dbscan = DBSCAN(eps=0.5, min_samples=1, metric='cosine')  # Adjust eps and min_samples as needed
# clusters = dbscan.fit_predict(embeddings)

# Calculate the silhouette score
silhouette_avg = silhouette_score(embeddings, clusters, metric='cosine')
print(f'Silhouette Score: {silhouette_avg}')

# Calculate the Davies-Bouldin index
db_index = davies_bouldin_score(embeddings, clusters)
print(f'Davies-Bouldin Index: {db_index}')
# breakpoint()

# Create a defaultdict with lists as default values
cluster_dict = defaultdict(list)

# Populate the dictionary
for string, cluster_id in zip(commands, clusters):
    cluster_dict[cluster_id].append(string)

# Convert defaultdict to a regular dictionary (optional)
cluster_dict = dict(cluster_dict)

# Find the cluster with the longest list
sorted_clusters = sorted(cluster_dict, key=lambda k: len(cluster_dict[k]), reverse=True)

tot = 0
for i in range(1,101):
    print(f"{len(cluster_dict[sorted_clusters[-i]])} elements.")
    tot += len(cluster_dict[sorted_clusters[-i]])

print(f"{tot} total")

# Apply t-SNE for dimensionality reduction
reduced_embeddings = TSNE(n_components=2, metric='cosine', perplexity=175).fit_transform(embeddings)

# Plot the t-SNE results
plt.figure(figsize=(10, 8))
for cluster in np.unique(clusters):
    cluster_points = reduced_embeddings[clusters == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

    # Calculate the ellipse for each cluster
    if len(cluster_points) > 1:  # Ellipse requires at least 2 points
        cov = np.cov(cluster_points, rowvar=False)
        mean = np.mean(cluster_points, axis=0)

        # Eigen decomposition of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

        # Compute width, height and angle of ellipse
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)

        # Draw the ellipse
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='black', facecolor='none', lw=2)
        plt.gca().add_patch(ellipse)

plt.title('t-SNE with Cluster Ellipses')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.grid(True)
plt.show()