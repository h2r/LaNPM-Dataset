from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Mapping, Tuple
import h5py
import os
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import pickle

from ai2thor.server import MultiAgentEvent


from metrics.base_metric import \
    AreaCoverage, CLIP_SemanticUnderstanding, GoalDistanceDiff, Metric, \
    RootMSE, TrajData, Length, GraspSuccRate, DeltaDist
from metrics.task_succ import TaskSuccMetric

ap = ArgumentParser()
ap.add_argument("--gt_traj_path", type=str, default=os.environ['HOME'] + '/data/shared/lanmp/lanmp_dataset.hdf5')
ap.add_argument("--eval_traj_path", type=str, default="")
ap.add_argument("--use_gt_for_eval", action='store_true')
ap.add_argument("--save_csv_file", type=str, default=f"results/eval_{datetime.now().strftime('%m%d%Y_%H%M%S')}.csv")
ap.add_argument("--print_every_step", action='store_true')



class Evaluator():
    def __init__(self, gt_dataset_path: str):
        self.gt_file = h5py.File(gt_dataset_path, 'r')
        self.cmd_to_traj_name: Mapping[str, str] = {}
        self.scene_to_cmd: Mapping[str, List[str]] = defaultdict(list)
        self._build_traj_index()

        # initialize the metrics with pre-computed info
        self.metrics: List[Metric] = [
            # AreaCoverage(),
            CLIP_SemanticUnderstanding(scene_to_cmds=self.scene_to_cmd),
            RootMSE(),
            DeltaDist(),
            GoalDistanceDiff(),
            # TaskSuccMetric(),
            GraspSuccRate(),
            Length(),
        ]
    
    def _build_traj_index(self):
        self.cmd_to_traj_name = {}
        for traj_name, traj_data in self.gt_file.items():
            # breakpoint()
            json_data_step0 = json.loads(traj_data['folder_0'].attrs['metadata'])
            nl_cmd = json_data_step0['nl_command']
            scene = json_data_step0['scene']
            self.cmd_to_traj_name[nl_cmd] = traj_name
            self.scene_to_cmd[scene].append(nl_cmd)
    
    
    def read_eval_traj(self, traj_path) -> Tuple[TrajData, str, str, dict]:
        """
        Read a trajectory from a pickle file.
        """
        with open(traj_path, 'rb') as f:
            traj_data = pickle.load(f)
        scene = ""
        cmd = ""
        img_history = []
        xyz_body_history = []
        xyz_ee_history = []
        yaw_body_history = []
        error_history = []
        action_history = []
        num_steps = 0

        # keys are ['task', 'scene', 'img', 'xyz_body', 'xyz_body_delta', 'yaw_body', 'yaw_body_delta', 'pitch_body', 'xyz_ee', 'xyz_ee_delta', 'pickup_dropoff', 'holding_obj', 'control_mode', 'action', 'terminate', 'step', 'timeout', 'error']
        for entry in traj_data['trajectory_data']:
            scene = entry['scene']
            cmd = entry['task']
            
            while len(entry['img'].shape) > 3:
                entry['img'] = entry['img'][0]
            img_history.append(entry['img'])
            
            xyz_body_history.append(entry['xyz_body'])
            yaw_body_history.append(entry['yaw_body'])
            xyz_ee_history.append(entry['xyz_ee'])
            
            if type(entry['error']) == list:
                entry['error'] = entry['error'][0]
            error_history.append(entry['error'])
            action_history.append(entry['action'])
            num_steps += 1
        
        return TrajData(
            img=np.array(img_history), xyz_body=np.array(xyz_body_history), yaw_body=np.array(yaw_body_history),
            xyz_ee=np.array(xyz_ee_history), steps=num_steps,
            errors=error_history, action=action_history
        ), cmd, scene, traj_data['final_state']

            
    
    def convert_gt_hdf5_entry(self, traj_hdf_group: h5py.Group, desired_len: int) -> Tuple[TrajData, str, str]:
        img_history = []
        xyz_body_history = []
        xyz_ee_history = []
        yaw_body_history = []
        error_history = []
        action_history = []
        num_steps = 0

        nl_command = ""
        scene = ""

        for timestep_name, timestep_group in traj_hdf_group.items():
            if 'metadata' in timestep_group.attrs:
                json_data_serialized = timestep_group.attrs['metadata']
                # Deserialize the JSON string into a Python dictionary
                json_data = json.loads(json_data_serialized)
                # Access 'steps' from the dictionary
                step_info = json_data['steps']
                nl_command = json_data['nl_command']
                scene = json_data['scene']
                action = step_info[0]['action']

                img_key = [key for key in timestep_group.keys() if key.startswith("rgb_")]
                img: np.ndarray = timestep_group[img_key[0]][()] # 720x1080x3
                state_body = step_info[0]['state_body'][:3]
                body_yaw = step_info[0]['state_body'][-1]
                state_ee = step_info[0]['state_ee'][:3]
                
                xyz_body_history.append(state_body)
                yaw_body_history.append(body_yaw)
                xyz_ee_history.append(state_ee)
                img_history.append(img)
                action_history.append(action)
                error_history.append(None)
                
                num_steps += 1
        original_num_steps = num_steps
        while num_steps < desired_len:
            xyz_body_history.append(state_body)
            yaw_body_history.append(body_yaw)
            xyz_ee_history.append(state_ee)
            img_history.append(img)
            action_history.append(action)
            error_history.append(None)
            num_steps += 1
        
        return TrajData(
            img=np.array(img_history), xyz_body=np.array(xyz_body_history), yaw_body=np.array(yaw_body_history),
            xyz_ee=np.array(xyz_ee_history), steps=original_num_steps,
            errors=error_history, action=action_history
        ), nl_command, scene


    def _pad_eval_traj(self, eval_traj: TrajData, desired_length: int):
        img_history = list(eval_traj.img)
        xyz_body_history = list(eval_traj.xyz_body)
        yaw_body_history = list(eval_traj.yaw_body)
        xyz_ee_history = list(eval_traj.xyz_ee)
        action_history = eval_traj.action
        error_history = eval_traj.errors
        num_steps = eval_traj.steps

        while num_steps < desired_length:
            xyz_body_history.append(xyz_body_history[-1])
            yaw_body_history.append(yaw_body_history[-1])
            xyz_ee_history.append(xyz_ee_history[-1])
            img_history.append(img_history[-1])
            action_history.append(action_history[-1])
            error_history.append(None)
            num_steps += 1
        return TrajData(
            img=np.array(img_history), xyz_body=np.array(xyz_body_history), yaw_body=np.array(yaw_body_history),
            xyz_ee=np.array(xyz_ee_history), steps=num_steps,
            errors=error_history, action=action_history
        )

    
    def evaluate_one_traj(self, scene: str, cmd: str, exec_traj: TrajData, end_inf_state: MultiAgentEvent, pad_eval=True) -> Mapping[str, float]:
        # load gt traj
        gt_traj_name = self.cmd_to_traj_name[cmd]
        gt_traj_h5 = self.gt_file[gt_traj_name]
        gt_desired_len = len(exec_traj.errors)

        # convert hdf5 to data. 
        # padded gt length
        gt_traj, cmd2, scene2 = self.convert_gt_hdf5_entry(gt_traj_h5, gt_desired_len)

        assert cmd2 == cmd, "Command mismatch for eval."
        assert scene2 == scene, "Scene name mismatch for eval."

        # padded eval length if needed. otherwise cut down gt traj
        if pad_eval:
            # pad exec traj with the last time step
            if len(exec_traj.errors) < len(gt_traj.errors):
                exec_traj = self._pad_eval_traj(exec_traj, len(gt_traj.errors))
        else:
            # cut gt traj to match exec traj len. TODO test
            for key in gt_traj.__dict__.keys():
                if type(getattr(gt_traj, key)) in [list, np.ndarray]:
                    gt_traj_h5[key] = gt_traj[key][:len(exec_traj.errors)]

        # get results
        results_dict = {}
        for metric in self.metrics:
            score = metric.get_score(scene, exec_traj, gt_traj, end_inf_state, cmd)
            if type(score) == dict:
                for key, val in score.items():
                    results_dict[f"{metric.name}/{key}"] = val
            else:
                results_dict[metric.name] = score
        return results_dict

    def __del__(self):
        self.gt_file.close()


def main():
    args = ap.parse_args()
    evaluator = Evaluator(args.gt_traj_path)
    if args.use_gt_for_eval:
        eval_gt(evaluator, args.gt_traj_path, args.save_csv_file, print_every_step=args.print_every_step)
    else:
        eval_model_traj(evaluator, args.eval_traj_path, args.save_csv_file, print_every_step=args.print_every_step)


def ensure_path_exists(filename: str):
    # src: https://stackoverflow.com/questions/12517451
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def eval_model_traj(evaluator: Evaluator, traj_path: str, save_csv_file: str, print_every_step: bool=False):
    """
    Run evaluation on a model trajectory.
    """
    result_list = []
    files = os.listdir(traj_path)
    for file in tqdm(files):
        file_name = os.path.join(traj_path, file)
        print(file_name)
        traj_data, cmd, scene, end_inf_state = evaluator.read_eval_traj(file_name)
        breakpoint()
        result = evaluator.evaluate_one_traj(scene, cmd, traj_data, end_inf_state)
        result['cmd'] = cmd
        result['scene'] = scene
        result['model_name'] = "model"
        result_list.append(result)
        if print_every_step: 
            print(result)

    df = pd.DataFrame(result_list)
    ensure_path_exists(save_csv_file)
    df.to_csv(save_csv_file, index=False)
    print("Results written to", save_csv_file)

def eval_gt(evaluator: Evaluator, gt_path: str, save_csv_file: str, print_every_step: bool=False):
    """
    Run evaluation on ground truth dataset.
    """
    result_list = []
    with h5py.File(gt_path, 'r') as hdf_file:
        # count = -1 # skip ahead
        for traj_name, traj_content in tqdm(hdf_file.items()):
            # count += 1 # skip ahead
            # if count < 23: continue # skip ahead
            converted_traj, cmd, scene = evaluator.convert_gt_hdf5_entry(traj_content, len(traj_content.keys()))
            result = evaluator.evaluate_one_traj(scene, cmd, converted_traj, None)
            result['cmd'] = cmd
            result['scene'] = scene
            result['model_name'] = "gt"
            result_list.append(result)
            if print_every_step: 
                print(result)

    df = pd.DataFrame(result_list)
    ensure_path_exists(save_csv_file)
    df.to_csv(save_csv_file, index=False)
    print("Results written to", save_csv_file)


if __name__ == "__main__":
    main()

