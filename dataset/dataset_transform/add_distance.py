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

# with open("scene_objs.json", "r") as json_file:
#     scene_objs_json = json.load(json_file)

# obj = extract_obj("Get the book on the desk and put it on the shelf.", scene_objs_json, "FloorPlan_Train7_5")
# print(obj)
# breakpoint()
# exit()

scene_objs= {}
scene_objs_json = None

PATH = "/users/ajaafar/data/shared/lanmp/sim_dataset.hdf5" #HDF5 dataset path

with h5py.File(PATH, 'r') as hdf_file:
    # Iterate through each trajectory group
    for trajectory_name, trajectory_group in hdf_file.items():
        print(f"Trajectory: {trajectory_name}")
        # Iterate through each timestep group within the trajectory
        for timestep_name, timestep_group in trajectory_group.items():
            # print(f"  Step: {timestep_name}")

            # Read and decode the JSON metadata
            metadata = json.loads(timestep_group.attrs['metadata'])
            # print(f"Metadata: {json.dumps(metadata, indent=2)}")
            scene = metadata['scene']
            event = get_event(scene)
            if not os.path.exists('scene_objs.json'):
                if scene not in scene_objs.keys():
                    scene_objs[scene] = get_all_objs(scene)

                if len(scene_objs.keys()) == 5 and not os.path.exists('scene_objs.json'):
                    with open("scene_objs.json", "w") as json_file:
                        json.dump(scene_objs, json_file, indent=4)
            else:
                if scene_objs_json == None:
                    with open("scene_objs.json", "r") as json_file:
                        scene_objs_json = json.load(json_file)
            
            command = metadata['nl_command']
            obj = extract_obj(command, scene_objs_json, scene)
            obj_pos = get_obj_pos(event, obj)
            ee_pos = metadata['steps'][0]['state_ee'][:3]

            dist = np.linalg.norm(np.array(list(obj_pos.values())) - np.array(ee_pos))
            breakpoint()
            print('hi')