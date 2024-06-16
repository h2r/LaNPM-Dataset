import transformations as tft
from math import degrees
from datetime import datetime
import logging
import numpy as np
import json
import zipfile
from io import BytesIO

num = 0

class get_data():
    def __init__(self):
        self.data = [] #each point is a step
        self.rgb = []
        self.depth = []
        self.inst_seg = []
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
        global num

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
        # dic['rgb'] = event.frame
        # dic['depth'] = event.depth_frame
        # dic['inst_seg'] = event.instance_segmentation_frame
        dic['inst_det2D'] = {'keys': list(event.instance_detections2D.keys()), 'values': list(event.instance_detections2D.values())}
        dic['rgb'] = f'./rgb_{num}.npy'
        dic['depth'] = f'./depth_{num}.npy'
        dic['inst_seg'] = f'./inst_seg_{num}.npy'
        num+=1


        if self.prev_state_body == dic['state_body'] and self.prev_state_ee == dic['state_ee'] and self.prev_hand_sphere_center == dic['hand_sphere_center'] and self.prev_held_objs == dic['held_objs'] and self.prev_held_objs_state == dic['held_objs_state']:
            return
        
        self.prev_state_body = dic['state_body']
        self.prev_state_ee = dic['state_ee']
        self.prev_hand_sphere_center = dic['hand_sphere_center']
        self.prev_held_objs = dic['held_objs']
        self.prev_held_objs_state = dic['held_objs_state']

        self.data.append(dic)
        self.rgb.append(event.frame)
        self.depth.append(event.depth_frame)
        self.inst_seg.append(event.instance_segmentation_frame)


    # def _ndarray_to_list(self,obj):
    #         logging.info("Processing an object of type %s", type(obj))
    #         if isinstance(obj, dict):
    #             return {key: self._ndarray_to_list(value) if key in ['inst_seg', 'depth', 'rgb', 'inst_det2D'] or isinstance(value, (dict, list)) else value
    #                     for key, value in obj.items()}
    #         elif isinstance(obj, list):
    #             return [self._ndarray_to_list(element) for element in obj]
    #         elif isinstance(obj, np.ndarray):
    #             return obj.tolist()
    #         else:
    #             return obj


    def _chunk_dict(self, data, chunk_size):
        """Yield successive chunk_size chunks from the dictionary."""
        
        # Construct the base dictionary without the later keys and their associated lists
        #base_dict = {k: v for k, v in data.items() if k != 'steps'}
        list_data = data['steps']




        # Getting all keys if not equal to steps
        base_keys = [k for k in data.keys() if k != 'steps']

        # Iterate over the list in chunks
        for i in range(0, len(list_data), chunk_size):
            #chunked_dict = base_dict.copy()  # Copy the base data (without the large list)
            chunked_dict = {k: data[k] for k in base_keys}
            chunked_dict['steps'] = list_data[i:i+chunk_size]
            yield chunked_dict

        

    def save(self,command):
        final_dic = {"nl_command": command, "scene":self.scene, "steps":self.data}
        # final_dic = self._ndarray_to_list(final_dic)

        # with open('/home/user/NPM-Dataset/data.json', 'w') as f:
        #     json.dump(final_dic, f, indent=4)

        print("\nSaving...\n")
        CHUNK_SIZE=1
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        # Write the JSON data chunks directly into a zip file
        with zipfile.ZipFile(f'data_{current_time}.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
            for idx, chunk in enumerate(self._chunk_dict(final_dic, CHUNK_SIZE)):
                folder_name = f'folder_{idx}/'
                json_filename = f'{folder_name}data_chunk_{idx}.json'
                rgb_filename = f'{folder_name}rgb_{idx}.npy'
                depth_filename = f'{folder_name}depth_{idx}.npy'
                inst_seg_filename = f'{folder_name}inst_seg_{idx}.npy'


                with zipf.open(json_filename, 'w') as json_file:
                    # Convert the chunk to a JSON string and encode it to bytes
                    json_data = json.dumps(chunk).encode('utf-8')
                    json_file.write(json_data)

                # Define a helper function to save numpy data
                def save_npy_to_zip(data, filename):
                    npy_data = BytesIO()
                    np.save(npy_data, data, allow_pickle=True, fix_imports=True)
                    npy_data.seek(0)  # Rewind the buffer
                    with zipf.open(filename, 'w') as npy_file:
                        npy_file.write(npy_data.getvalue())
    
                save_npy_to_zip(self.rgb, rgb_filename)
                save_npy_to_zip(self.depth, depth_filename)
                save_npy_to_zip(self.inst_seg, inst_seg_filename)