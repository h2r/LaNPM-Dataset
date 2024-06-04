import os
import json
from typing import List
import numpy as np
from PIL import Image
from datetime import datetime
from eval import Eval
from env.thor_env import ThorEnv
import h5py

from metrics.task_succ import extract_task_succ, TaskSuccMetric
from metrics.base_metric import CLIP_SemanticUnderstanding, RootMSE, AreaCoverage, Metric, TrajData

class EvalTask(Eval):
    '''
    evaluate overall task performance
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv(args.pp_data)

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                traj = model.load_task_json(task)
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, resnet, traj, args, lock, successes, failures, results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()


    @classmethod
    def evaluate(cls, env, model, resnet, traj_data, args, lock, successes, failures, results):
        # reset model
        model.reset()
        # setup scene
        cls.setup_scene(env, traj_data, args)

        # extract language features
        feat = model.featurize([traj_data])

        # goal instr
        goal_instr = traj_data['ann']['task_desc']

        done, success = False, False
        fails = 0
        t = 0
        while not done:
            print(f"Step: {t} ", end="\r")

            # break if max_steps reached
            if t >= args.max_steps:
            # if t >= 3:
                break

            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            feat['frames'] = resnet.featurize([curr_image], batch=1).unsqueeze(0)

            # forward model
            m_out = model.step(feat, t)
            m_pred = model.extract_preds(m_out, [traj_data], feat, clean_special_tokens=False)
            m_pred = list(m_pred.values())[0]


            # # check if <<stop>> was predicted
            if m_pred['action_low_word'] == "stop":
                print("\tpredicted STOP")
                break

            # get action
            word_action = m_pred['action_low_word']
            num_action = m_pred['action_low_num']

            # print action
            if args.debug:
                print(action)

            #  use predicted action to interact with the env
            t_success, error, end_inf_state = env.take_action(word_action, num_action, args.rand_agent) # t_success: True, False, or None, or "stop"
            if t_success == "stop": # only for random agent
                print("\tpredicted STOP")
                break
            # optional
            # if t_success == False:
            #     fails += 1
            #     if fails >= args.max_fails:
            #         print("Interact API failed %d times" % fails)
            #         break
            t += 1

        exec_traj = None # TODO fill in
        gt_traj_name = traj_data['root'].rsplit('/', 1)[1]
        gt_traj, lang, scene = cls.get_gt_traj(gt_traj_name)
        results = cls.calc_metrics(exec_traj, gt_traj, end_inf_state, lang, scene)

        # lock.release()


    @classmethod
    def get_gt_traj(cls, gt_traj_name, desired_step_len=0):
        hdf5_file_path = os.environ['HOME'] + '/data/shared/lanmp/lanmp_dataset.hdf5' #TODO make relative later
        # Trajectory to fetch actions from
        trajectory_name = gt_traj_name

        traj_action_lst = []
        nl_command = None
        scene = None
        with h5py.File(hdf5_file_path, 'r') as hdf_file:
            if trajectory_name in hdf_file:
                # Access the group corresponding to the trajectory
                trajectory_group = hdf_file[trajectory_name]
                # Loop through each timestep group within the trajectory group

                img_history = []
                xyz_body_history = []
                xyz_ee_history = []
                yaw_body_history = []
                steps = 0

                for timestep_name, timestep_group in trajectory_group.items():
                    if 'metadata' in timestep_group.attrs:
                        json_data_serialized = timestep_group.attrs['metadata']
                        # Deserialize the JSON string into a Python dictionary
                        json_data = json.loads(json_data_serialized)
                        # Access 'steps' from the dictionary
                        steps = json_data['steps']
                        nl_command = json_data['nl_command']
                        scene = json_data['scene']
                        action = steps[0]['action']

                        img_key = [key for key in timestep_group.keys() if key.startswith("rgb_")]
                        img: np.ndarray = timestep_group[img_key[0]][()] # 720x1080x3
                        state_body = steps[0]['state_body'][:3]
                        body_yaw = steps[0]['state_body'][-1]
                        state_ee = steps[0]['state_ee'][:3]
                        
                        xyz_body_history.append(state_body)
                        yaw_body_history.append(body_yaw)
                        xyz_ee_history.append(state_ee)
                        img_history.append(img)
                        
                        steps += 1
                while steps < desired_step_len:
                    xyz_body_history.append(state_body)
                    yaw_body_history.append(body_yaw)
                    xyz_ee_history.append(state_ee)
                    img_history.append(img)
        
        return TrajData(
            img=np.array(img_history), xyz_body=np.array(xyz_body_history), yaw_body=np.array(yaw_body_history),
            xyz_ee=np.array(xyz_ee_history), steps=steps
        ), nl_command, scene

    @classmethod
    def calc_metrics(cls, exec_traj: TrajData, gt_traj: TrajData, end_inf_state: dict, lang: str, scene: str):
        """
        
        Temporary wrapper method that calls Yichen's method that gets all the metric results. 
        Made it a wrapper so it's more isolated and easier for Yichen to look at. 
        Later, I will remove the wrapper and call his method directly in my main inference code
        gt_traj is a single ground truth trajectory. It is a list of dicts where each dict has the action, xyz of body, yaw of body, and xyz of ee for a timestep. xyz are global coords. yaw is degrees
        end_inf_state is the end/last return from the controller from inference episode that has all robot and env info. If this is None, that means the robot didn't move at all and the episode ended, which either shouldn't happen or rarely happen so don't worry about it too much
        lang is the NL command
        scene is the name of the trajectory's scene
        
        """
        extract_task_succ()

        metrics: List[Metric] = [
            AreaCoverage(),
            CLIP_SemanticUnderstanding(),
            RootMSE(),
            TaskSuccMetric()
        ]

        results_dict = {
            metric.name: metric.get_score(scene, exec_traj, gt_traj, end_inf_state, lang) for metric in metrics
        }
        results_dict['length'] = exec_traj.steps
        breakpoint()
        return results_dict


    def create_stats(self):
            '''
            storage for success, failure, and results info
            '''
            self.successes, self.failures = self.manager.list(), self.manager.list()
            self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'task_results_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)
