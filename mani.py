from ai2thor.controller import Controller
import time
import curses
from PIL import Image
import numpy as np
import transformations as tft
from math import degrees
import json
from datetime import datetime


import ai2thor

# data = [] #each point is a step
# scene = ''

# def convert_to_euler(rot_quat):

#     # Extract quaternion from the dictionary
#     quaternion = [rot_quat['x'], rot_quat['y'], rot_quat['z'], rot_quat['w']]

#     # Convert quaternion to Euler angles
#     roll, pitch, yaw = tft.euler_from_quaternion(quaternion)

#     roll = degrees(roll)
#     pitch = degrees(pitch)
#     yaw = degrees(yaw)

#     return (roll, pitch, yaw)

# def gather_data(event):
#     global scene
#     scene = event.metadata['sceneName']

#     dic = {}
#     dic['timestamp'] = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]  # Truncate to milliseconds
#     # dic['rgb'] = event.frame
#     # dic['depth'] = event.depth_frame
#     # dic['seg'] = event.instance_segment.0. for the user input
#     global_coord_agent = event.metadata['agent']['position']
#     yaw_agent =  event.metadata['agent']['rotation']['y'] #degrees
#     dic['state_body'] = (global_coord_agent['x'], global_coord_agent['y'], global_coord_agent['z'], yaw_agent)
#     # global_coord_ee = event.metadata["arm"]["joints"][3]['position']
#     # rot_ee = convert_to_euler(event.metadata["arm"]["joints"][3]['rotation'])
#     # dic['state_ee'] = (global_coord_ee['x'], global_coord_ee['y'],global_coord_ee['z'], rot_ee[0], rot_ee[1], rot_ee[2])
#     data.append(dic)
#     print("data: ", data)



if __name__ == "__main__":
    controller = Controller(
        agentMode="arm",
        # agentMode="default",
        # massThreshold=None,
        scene="FloorPlan_Train3_1",
        snapToGrid=False,
        visibilityDistance=1.5,
        gridSize=0.25,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        width= 1280,
        height= 720,
        fieldOfView=60
    )

    while True:            
        controller.interact(metadata=True, color_frame=True, depth_frame=True, semantic_segmentation_frame=True, instance_segmentation_frame=True)

# final_dic = {"nl_command": "blah blah blah", "scene":scene, "steps":data}
# print(final_dic)

# import logging
# def ndarray_to_list(obj):
#     logging.info("Processing an object of type %s", type(obj))
#     if isinstance(obj, dict):
#         return {key: ndarray_to_list(value) if key in ['seg', 'depth', 'rgb'] or isinstance(value, (dict, list)) else value
#                 for key, value in obj.items()}
#     elif isinstance(obj, list):
#         return [ndarray_to_list(element) for element in obj]
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     else:
#         return obj


# final_dic = ndarray_to_list(final_dic)

# with open('data.json', 'w') as f:
#     json.dump(final_dic, f, indent=4)

# depth_frame = final_dic["steps"][-1]["d"]

# depth_frame_normalized = ((depth_frame - depth_frame.min()) * (1/(depth_frame.max() - depth_frame.min()) * 255)).astype(np.uint8)

# Convert to PIL Image and show
# image = Image.fromarray(depth_frame_normalized)
# image = Image.fromarray(final_dic["steps"][-1]["seg"])

# image.show()