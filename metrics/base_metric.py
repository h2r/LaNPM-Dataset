from typing import List, Mapping, Union
from dataclasses import dataclass
import numpy as np

from ai2thor.controller import Controller
import torch
import clip
from PIL import Image
import h5py
import json

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

@dataclass
class TrajData:
    """
    Data for a single trajectory, contain a sequence of images, body position, body yaw, and end effector position,
    each represented as a numpy array of size n x [...] where n is the length of the trajs.
    """
    img: np.ndarray # nxhxwx3
    xyz_body: np.ndarray # nx3, global position of the body
    yaw_body: np.ndarray # nx1
    xyz_ee: np.ndarray # nx3, global position of the end effector
    errors: List[Union[None, str]] # list of error messages (str), indexed by the time step they're happening.
    action: List[str] # list of str actions
    steps: np.int32 #integer, total unpadded number of steps in the trajectory


class Metric:
    def __init__(self, name):
        self.name = name

    def get_score(
            self, 
            scene_name: str, 
            traj_model: TrajData, 
            traj_gt: TrajData,
            final_state: Controller, 
            task_cmd: str
        ) -> Union[float, Mapping[str, float]]:
        raise NotImplementedError("Subclasses should implement this!")




class RootMSE(Metric):

    def __init__(self, name = 'rmse', weightage = [0.33, 0.33, 0.34]):
        self.name = name
        self.property_weights = weightage

    def get_score(self, scene_name: str, traj_model: TrajData, traj_gt: TrajData, final_state: Controller, task_cmd: str):
        
        score = {}

        # assert(len(traj_model[traj_model.keys()[0]]) == len(traj_gt[traj_gt.keys()[0]]), "Error: GT and model traj don't have the same length")
        assert len(dir(traj_model)) == len(dir(traj_gt)), "Error: GT and model traj don't have the same keys"

        traj_model_data = {'xyz_body': traj_model.xyz_body, 'yaw_body': traj_model.yaw_body, 'xyz_ee': traj_model.xyz_ee}
        traj_gt_data = {'xyz_body': traj_gt.xyz_body, 'yaw_body':traj_gt.yaw_body, 'xyz_ee': traj_gt.xyz_ee}

        for key in traj_model_data.keys():

            model_data = np.array(traj_model_data[key])
            gt_data = np.array(traj_gt_data[key])



            mse = np.square(model_data - gt_data)

            if len(mse.shape) > 1 and mse.shape[1] > 1:
                mse = np.sum(mse, axis=1)
            mse = np.mean(mse, axis=0)
            rmse = np.sqrt(mse)

            score[key] = rmse
        

        weighted_score = 0

        for weight, key in zip(self.property_weights, score.keys()):

            weighted_score += weight * score[key]
        
        score['overall_weighted'] = weighted_score
    
        return score


class EndDistanceDiff(Metric):
    """
    Computes the distance between the gt last epi and the true last epi xyz
    """
    def __init__(self, name="distance_diff", diff_type="body"):
        assert diff_type in ["body", "ee"], diff_type + " distance type not implemented. implemented options are: body, ee"
        self.diff_type = diff_type
        self.name = name
    
    def get_score(self, scene_name: str, traj_model: TrajData, traj_gt: TrajData, final_state: Controller, task_cmd: str):
        if self.diff_type == "body":
            abs_diff = traj_model.xyz_body[-1] - traj_gt.xyz_body[-1]
        elif self.diff_type == "ee":
            abs_diff = traj_model.xyz_ee[-1] - traj_gt.xyz_ee[-1]
        return np.sqrt(np.mean(abs_diff ** 2))


class GraspSuccRate(Metric):
    '''
    Computes the success rate of grasping
    '''
    def __init__(self, name = 'grasp_succ_rate'):
        self.name = name

    def get_score(self, scene_name: str, traj_model: TrajData, traj_gt: TrajData, final_state: Controller, task_cmd: str):
        # TODO check if error will be thrown if nothing to pickup
        total_mani_count = 0
        total_mani_errors = 0
        for action, error_msg in zip(traj_model.action, traj_model.errors):
            if action in ["PickupObject", "PutObject"]:
                total_mani_count += 1
                if error_msg is not None:
                    total_mani_errors += 1
        return 1 - total_mani_errors / total_mani_count if total_mani_count > 0 else np.nan


class Length(Metric):
    '''
    Computes the length of the episode
    '''
    def __init__(self, name = 'length'):

        self.name = name
    

    def get_score(self, scene_name: str, traj_model: TrajData, traj_gt: TrajData, final_state: Controller, task_cmd: str):
        return traj_model.steps


class AreaCoverage(Metric):

    '''
    Computes the area formed by the convex hull of all points the robot cross during the rollout
    We can compute the area normalized by the total number of steps the agent takes
    '''

    def __init__(self, name = 'area_coverage'):

        self.name = name
    

    def get_score(self, scene_name: str, traj_model: TrajData, traj_gt: TrajData, final_state: Controller, task_cmd: str):
        # Compute the convex hull
        hull = ConvexHull(traj_model.xyz_body[:,:2])

        # Get the vertices of the convex hull
        hull_points = traj_model.xyz_body[hull.vertices]

        # Create a polygon from the hull points
        polygon = Polygon(hull_points)

        # Compute the area of the polygon
        traversed_area = polygon.area

        #compute area coverage per unit of trajectory
        score = traversed_area / len(traj_model.xyz_body)

        return score




class CLIP_SemanticUnderstanding(Metric):

    def __init__(
            self, 
            name='clip_semantic_understanding', 
            bellman_lambda=0.99, 
            ema_interval=10,
            dataset_path=None
        ):

        '''
        1. Bellman like discounted reward equation
        2. Exponential moving average
        3. Task v.s. other task prediction
        '''
        self.name = name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=self.device)
        
        #hyperparams for the CLIP metric
        self.bellman_lambda = bellman_lambda
        self.ema_alpha = 2 / (ema_interval + 1)
        self.ema_interval = ema_interval

        #list of all tasks to be performed in the current scene
        self.scene_to_keys = self.split_by_scene(dataset_path)
        self.hdf =  h5py.File(dataset_path, 'r')
        



    def split_by_scene(self, hdf5_path):

        #mapping which keys are relevant to specific scenes
        scene_to_keys = {}

        with h5py.File(hdf5_path, 'r') as hdf_file:

            keys = list(hdf_file.keys())

            for k in keys:
                traj_json_dict = json.loads(hdf_file[k]['folder_0'].attrs['metadata'])

                if traj_json_dict['scene'] not in scene_to_keys:
                    scene_to_keys[traj_json_dict['scene']] = []
                
                scene_to_keys[traj_json_dict['scene']].append(k)
        
        for k in scene_to_keys.keys():
            scene_to_keys[k] = list(sorted(scene_to_keys[k]))
        
        with open('./lanmp_dataloader_scene_to_keys.json', 'w') as f:
            json.dump(scene_to_keys, f)

        return scene_to_keys


    def get_score(self, scene_name: str, traj_model: TrajData, traj_gt: TrajData, final_state: Controller, task_cmd: str):

        
        # assert(len(traj_model[traj_model.keys()[0]]) == len(traj_gt[traj_gt.keys()[0]]), "Error: GT and model traj don't have the same length")
        assert 'img' in dir(traj_model) and 'img' in dir(traj_gt), "Error: image key not in model or gt trajectory"
        

        discounted_clip_reward = 0.0
        ema_clip_reward = 0.0


        for i in range(traj_model.img.shape[0]):

            preprocessed_image = self.clip_preprocess(Image.fromarray(traj_model.img[i])).unsqueeze(0).to(self.device)

            preprocessed_text = clip.tokenize([traj_model.task_cmd]).to(self.device)

            with torch.no_grad():

                image_features = self.clip_model.encode_image(preprocessed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                text_features = self.clip_model.encode_text(preprocessed_text)
                text_features /= text_features.norm(dim=-1, keepdim=True)


                clip_similarity_score = image_features @ text_features.T
            


            discounted_clip_reward += self.bellman_lambda*clip_similarity_score if i < len(traj_model.img.shape[0])-1 else clip_similarity_score

            if (i+1) <= self.ema_interval:
                ema_clip_reward += clip_similarity_score / self.ema_interval
            else:
                ema_clip_reward = self.ema_alpha*clip_similarity_score + (1-self.ema_alpha)*ema_clip_reward







        scene_keys = self.scene_to_keys[scene_name]
        all_tasks_for_scene = set([])

        for task in scene_keys:
            traj_group = self.hdf[task]
            traj_steps = list(traj_group.keys())

            json_str = traj_group[traj_steps[0]].attrs['metadata']
            traj_json_dict = json.loads(json_str)
            nl_command = traj_json_dict['nl_command']

            all_tasks_for_scene.add(nl_command)

        all_tasks_for_scene = list(all_tasks_for_scene)
        success_index = all_tasks_for_scene.index(traj_model.task_cmd)

        correct_task_clip_score = 0

        for i in range(len(traj_model.img.shape[0])):

            preprocessed_image = self.clip_preprocess(Image.fromarray(traj_model.img[i])).unsqueeze(0).to(self.device)

            preprocessed_text = clip.tokenize(all_tasks_for_scene).to(self.device)

            with torch.no_grad():

                image_features = self.clip_model.encode_image(preprocessed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                text_features = self.clip_model.encode_text(preprocessed_text)
                text_features /= text_features.norm(dim=-1, keepdim=True)


                clip_similarity_score = image_features @ text_features.T
            
            if np.argmax(clip_similarity_score, axis=1) == success_index:
                correct_task_clip_score += 1
        

        correct_task_clip_score = correct_task_clip_score / len(traj_model.img.shape[0])




        scores = {'ema_clip_reward': ema_clip_reward, 'discounted_clip_reward': discounted_clip_reward, 'correct_task_clip_score': correct_task_clip_score}
        return scores
