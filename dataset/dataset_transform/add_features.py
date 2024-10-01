import json
import h5py
from ai2thor.controller import Controller
import os
import openai
import numpy as np

def get_event(scene):
    controller = Controller(
        agentMode="locobot",
        visibilityDistance=1.5,
        scene=scene,
        gridSize=0.25,
        movementGaussianSigma=0.005,
        rotateStepDegrees=90,
        rotateGaussianSigma=0.5,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=300,
        height=300,
        fieldOfView=60
    )
    event = controller.step("RotateLeft")
    controller.stop()
    return event

def get_all_objs(scene):
    event = get_event(scene)
    obj_lst_raw = event.metadata["objects"]
    all_objs = []
    for dic in obj_lst_raw:
        obj = dic['name']
        all_objs.append(obj)

    return all_objs

def extract_obj(command, scene_objs_json, scene):

    openai.api_key = ''

    prompt = f"Given this sentence: '{command}'\n\n Please extract the object the is meant to be gotten / picked up. Then match it to the most similar/same object from this list (if there are multiple in the list, just pick any of them): {scene_objs_json[scene]}.\n\n For example if the extracted object is 'salt' then 'Salt_Shaker_1' would be the most similar object from the list. Then please return that chosen object from the list as the output. Only return that as your answer and nothing else. No other words. No punctuation."

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You are a master word extractor. You know daily objects very well. You know how to match words."},
            {"role": "user", "content": prompt},
        ]
    )

    output_obj = response['choices'][0]['message']['content']

    return output_obj

def get_obj_pos(event, obj):
    obj_lst_raw = event.metadata["objects"]
    for dic in obj_lst_raw:
        if dic['name'] == obj:
            return dic['position']


scene_objs= {}
scene_objs_json = None

PATH = "/users/ajaafar/data/shared/lanmp/sim_dataset.hdf5" #HDF5 dataset path
NEW_PATH = "/users/ajaafar/data/shared/lanmp/sim_dataset_augmented.hdf5" #augmented HDF5 dataset path

# with h5py.File(PATH, 'r') as hdf_file:
with h5py.File(PATH, 'r') as hdf_file, h5py.File(NEW_PATH, 'w') as new_hdf_file:
    # Iterate through each trajectory group
    # dic = {}
    # traj_obj_dic = {}
    # obj_pos_dic = {}
    j = 1
    with open("traj_obj_dic.json", "r") as json_file:
    	traj_obj_dic = json.load(json_file)
    with open("traj_goal_pos.json", "r") as json_file:
        traj_goal_pos = json.load(json_file)   
    with open("obj_pos_dic.json", "r") as json_file:
        obj_pos_dic = json.load(json_file)   

    for trajectory_name, trajectory_group in hdf_file.items():
        # print(f"Trajectory: {trajectory_name}")
        if trajectory_name == "data_20:07:12":
            continue
        i = 0
        # obj_pos = None
        new_trajectory_group = new_hdf_file.create_group(trajectory_name)

        # dist = np.inf
        # Iterate through each timestep group within the trajectory
        for timestep_name, timestep_group in trajectory_group.items():
            # print(f"  Step: {timestep_name}")
            new_timestep_group = new_trajectory_group.create_group(timestep_name)

            metadata = json.loads(timestep_group.attrs['metadata'])
            # action = metadata['steps'][0]['action']
            # if action == 'ReleaseObject' and dist < 0.9:
            #     if trajectory_name in dic.keys():
            #         dic[trajectory_name] += 1
            #     else:
            #         dic[trajectory_name] = 1
            
            # print(f"Metadata: {json.dumps(metadata, indent=2)}")
            scene = metadata['scene']
            # if not os.path.exists('scene_objs.json'):
            #     if scene not in scene_objs.keys():
            #         scene_objs[scene] = get_all_objs(scene)

            #     if len(scene_objs.keys()) == 5 and not os.path.exists('scene_objs.json'):
            #         with open("scene_objs.json", "w") as json_file:
            #             json.dump(scene_objs, json_file, indent=4)
            # else:
            #     if scene_objs_json == None: # so that the file is only opened and set once
            #         with open("scene_objs.json", "r") as json_file:
            #             scene_objs_json = json.load(json_file)
            
            # obj = traj_obj_dic[trajectory_name] #load obj from saved dict
            # event = get_event(scene)
            # obj_pos = get_obj_pos(event, obj)
            obj_pos = obj_pos_dic[trajectory_name]
            ee_pos = metadata['steps'][0]['state_ee'][:3]
            body_pos = metadata['steps'][0]['state_body'][:3]
            action = metadata['steps'][0]['action']
            held_objs = metadata['steps'][0]['held_objs']
            dist_to_object = np.linalg.norm(np.array(list(obj_pos.values())) - np.array(ee_pos))
            dist_to_goal = np.linalg.norm(np.array(traj_goal_pos[trajectory_name][:3] - np.array(body_pos)))
            if i == 0:
                # command = metadata['nl_command']
                # obj = extract_obj(command, scene_objs_json, scene)
                # traj_obj_dic[trajectory_name] = obj
                # obj = traj_obj_dic[trajectory_name] #load obj from saved dict
                # event = get_event(scene)    
                # obj_pos = get_obj_pos(event, obj)
                obj_pos_dic[trajectory_name] = obj_pos
                metadata['target_obj'] = traj_obj_dic[trajectory_name] 
                metadata['init_target_obj_pos'] = obj_pos
                metadata['goal_pos'] = traj_goal_pos[trajectory_name][:3]
                metadata["base_start_to_goal_dist"] = dist_to_goal
                metadata['ee_start_to_target_obj_dist'] = dist_to_object
            i+=1
            
            
            if action == "PickupObject" and not held_objs:
                metadata['steps'][0]['curr_ee_to_target_obj_dist'] = float(0)
            else:
                metadata['steps'][0]['curr_ee_to_target_obj_dist'] = dist_to_object

            metadata['steps'][0]['curr_base_to_goal_dist'] = dist_to_goal
            
            # NEW CODE START: Copy datasets and attributes to new timestep group
            for dataset_name, dataset in timestep_group.items():
                timestep_group.copy(dataset_name, new_timestep_group)

            # Store the updated metadata back into the new timestep group
            new_timestep_group.attrs['metadata'] = json.dumps(metadata)

            # Copy other attributes from the original timestep group to the new one
            for attr_name, attr_value in timestep_group.attrs.items():
                if attr_name != 'metadata':
                    new_timestep_group.attrs[attr_name] = attr_value
            # NEW CODE END

        print(f'Finished {j} trajectories...')
        j+=1
    # with open("dic_release.json", "w") as json_file:
    #     json.dump(dic, json_file, indent=4)
    # with open("traj_obj_dic.json", "w") as json_file:
    #     json.dump(traj_obj_dic, json_file, indent=4)
    # with open("obj_pos_dic.json", "w") as json_file:
    #     json.dump(obj_pos_dic, json_file, indent=4)
    print("FINISHED ALL!")
