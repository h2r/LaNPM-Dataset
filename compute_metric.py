from argparse import ArgumentParser
from typing import List, Mapping
import h5py
import os
import json
import numpy as np
from ai2thor.server import MultiAgentEvent

from metrics.base_metric import AreaCoverage, CLIP_SemanticUnderstanding, Metric, RootMSE, TrajData
from metrics.task_succ import TaskSuccMetric

ap = ArgumentParser()
ap.add_argument("--gt_traj_path", type=str, default=os.environ['HOME'] + '/data/shared/lanmp/lanmp_dataset.hdf5')
ap.add_argument("--eval_traj_path", type=str, default="")
ap.add_argument("--use_gt_for_eval", action='store_true')



class Evaluator():
    def __init__(self, gt_dataset_path):
        self.gt_file = h5py.File(gt_dataset_path, 'r')
        self.metrics: List[Metric] = [
            AreaCoverage(),
            CLIP_SemanticUnderstanding(dataset_path=gt_dataset_path),
            RootMSE(),
            TaskSuccMetric()
        ]
        self.cmd_to_traj_name: Mapping[str, str] = {}
        self.build_traj_index()
    
    def build_traj_index(self):
        self.cmd_to_traj_name = {}
        for traj_name, traj_data in self.gt_file.items():
            # breakpoint()
            json_data_step0 = json.loads(traj_data['folder_0'].attrs['metadata'])
            nl_cmd = json_data_step0['nl_command']
            self.cmd_to_traj_name[nl_cmd] = traj_name
    
    def convert_gt_hdf5_entry(self, traj_hdf_group, desired_len):
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
        while num_steps < desired_len:
            xyz_body_history.append(state_body)
            yaw_body_history.append(body_yaw)
            xyz_ee_history.append(state_ee)
            img_history.append(img)
        
        return TrajData(
            img=np.array(img_history), xyz_body=np.array(xyz_body_history), yaw_body=np.array(yaw_body_history),
            xyz_ee=np.array(xyz_ee_history), steps=num_steps,
            errors=error_history, action=action_history
        ), nl_command, scene

    
    def evaluate_one_traj(self, scene: str, cmd: str, exec_traj: TrajData, end_inf_state: MultiAgentEvent):
        gt_traj_name = self.cmd_to_traj_name[cmd]
        gt_traj = self.gt_file[gt_traj_name]
        exec_traj, cmd2, scene2 = self.convert_gt_hdf5_entry(gt_traj, len(exec_traj.errors))
        
        assert cmd2 == cmd, "Command mismatch for eval."
        assert scene2 == scene, "Scene name mismatch for eval."

        results_dict = {
            metric.name: metric.get_score(scene, exec_traj, gt_traj, end_inf_state, cmd) for metric in self.metrics
        }
        return results_dict

    def __del__(self):
        self.gt_file.close()


def main():
    args = ap.parse_args()
    evaluator = Evaluator(args.gt_traj_path)
    if args.use_gt_for_eval:
        eval_gt(evaluator, args.gt_traj_path)


def eval_gt(evaluator: Evaluator, gt_path):
    result_list = []
    with h5py.File(gt_path, 'r') as hdf_file:
        for traj_name, traj_content in hdf_file.items():
            converted_traj, cmd, scene = evaluator.convert_gt_hdf5_entry(traj_content, len(traj_content.keys()))
            result = evaluator.evaluate_one_traj(scene, cmd, converted_traj, None)
            result_list.append(result)
    print(result_list)



if __name__ == "__main__":
    main()

