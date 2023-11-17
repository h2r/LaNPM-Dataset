import transformations as tft
from math import degrees
from datetime import datetime
import logging
import numpy as np
import json
import zipfile


class get_data():
    def __init__(self):
        self.data = [] #each point is a step
        self.scene = ''
        self.prev_state_body = []
        self.prev_state_ee = []
        self.prev_hand_sphere_center = 10000000
        self.prev_held_objs = []
        self.prev_held_objs_state = {}

    def _convert_to_euler(self,rot_quat):

        # Extract quaternion from the dictionary
        quaternion = [rot_quat['x'], rot_quat['y'], rot_quat['z'], rot_quat['w']]

        # Convert quaternion to Euler angles
        roll, pitch, yaw = tft.euler_from_quaternion(quaternion)

        roll = degrees(roll)
        pitch = degrees(pitch)
        yaw = degrees(yaw)

        return (roll, pitch, yaw)

    def _get_objs_pos(self,held_objs, all_objs):
        pos_rot_dic = {}
        for held_obj_id in held_objs: #usually 1 held object
            for obj_dic in all_objs:
                if obj_dic['objectId'] == held_obj_id:
                    pos_rot_dic[held_obj_id] = {'position': obj_dic['position'],  'rotation':obj_dic['rotation']}
        return pos_rot_dic

    def gather_data(self, event):
        # global scene, prev_state_body, prev_state_ee, prev_hand_sphere_center, prev_held_objs, prev_held_objs_state
        self.scene = event.metadata['sceneName']

        dic = {}
        dic['sim_time'] = event.metadata['currentTime']
        dic['wall-clock_time'] = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
        dic['action'] = event.metadata['lastAction']
        global_coord_agent = event.metadata['agent']['position']
        yaw_agent =  event.metadata['agent']['rotation']['y'] #degrees
        dic['state_body'] = [global_coord_agent['x'], global_coord_agent['y'], global_coord_agent['z'], yaw_agent]
        global_coord_ee = event.metadata["arm"]["joints"][3]['position']
        rot_ee = self._convert_to_euler(event.metadata["arm"]["joints"][3]['rotation'])
        dic['state_ee'] = [global_coord_ee['x'], global_coord_ee['y'],global_coord_ee['z'], rot_ee[0], rot_ee[1], rot_ee[2]]
        dic['hand_sphere_center'] = event.metadata['arm']['handSphereRadius']
        dic['held_objs'] = event.metadata['arm']['heldObjects']
        pos_rot_dic = self._get_objs_pos(event.metadata['arm']['heldObjects'], event.metadata["objects"])
        dic['held_objs_state'] =  pos_rot_dic
        dic['rgb'] = event.frame
        dic['depth'] = event.depth_frame
        dic['inst_seg'] = event.instance_segmentation_frame

        if self.prev_state_body == dic['state_body'] and self.prev_state_ee == dic['state_ee'] and self.prev_hand_sphere_center == dic['hand_sphere_center'] and self.prev_held_objs == dic['held_objs'] and self.prev_held_objs_state == dic['held_objs_state']:
            return
        
        self.prev_state_body = dic['state_body']
        self.prev_state_ee = dic['state_ee']
        self.prev_hand_sphere_center = dic['hand_sphere_center']
        self.prev_held_objs = dic['held_objs']
        self.prev_held_objs_state = dic['held_objs_state']

        self.data.append(dic)

    def _ndarray_to_list(self,obj):
            logging.info("Processing an object of type %s", type(obj))
            if isinstance(obj, dict):
                return {key: self._ndarray_to_list(value) if key in ['inst_seg', 'depth', 'rgb'] or isinstance(value, (dict, list)) else value
                        for key, value in obj.items()}
            # Removing this elif statement since we want to convert to list anyways. Might save computation time
            # elif isinstance(obj, list): 
            #     return [self._ndarray_to_list(element) for element in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj


    def _chunk_dict(self, data, chunk_size):
        """Yield successive chunk_size chunks from the dictionary."""
        
        # Construct the base dictionary without the 'hi' key and its associated list
        # base_dict = {k: v for k, v in data.items() if k != 'steps'}
        list_data = data['steps']
        
        # Getting all keys if not equal to steps
        base_keys = [k for k in data.keys() if k != 'steps']
        

        # Iterate over the list in chunks
        for i in range(0, len(list_data), chunk_size):
         #   chunked_dict = base_dict.copy()  # Copy the base data (without the large list)
         
            # Should be more memory efficient since we dont make a separate copy of base_dict
            chunked_dict = {k: data[k] for k in base_keys} 
            chunked_dict['steps'] = list_data[i:i+chunk_size]
            yield chunked_dict

        


    def save(self):
        final_dic = {"nl_command": "Go to the bedroom, pick up the red apple, then go to the table which has the orange basketball and drop it on that table.", "scene":self.scene, "steps":self.data}
        final_dic = self._ndarray_to_list(final_dic)

        # with open('/home/user/NPM-Dataset/data.json', 'w') as f:
        #     json.dump(final_dic, f, indent=4)

        print("Saving...")
        CHUNK_SIZE=1
        # Write the JSON data chunks directly into a zip file
        with zipfile.ZipFile('data.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
            for idx, chunk in enumerate(self._chunk_dict(final_dic, CHUNK_SIZE)):
                with zipf.open(f'data_chunk_{idx}.json', 'w') as json_file:
                    # Convert the chunk to a JSON string and encode it to bytes
                    json_data = json.dumps(chunk).encode('utf-8')
                    json_file.write(json_data)