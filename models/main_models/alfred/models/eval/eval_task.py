import os
import json
import pickle
from typing import List, Tuple
import numpy as np
from PIL import Image
from datetime import datetime
from eval import Eval
from env.thor_env import ThorEnv
import h5py

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
        env.controller.stop()


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
        print('lang: ', goal_instr)

        print("root: ", traj_data['root'])
        print('scene: ', traj_data['scene'])

        done, success = False, False
        fails = 0
        t = 0
        
        end_inf_state = env.last_event.metadata
        end_inf_state_lst = [(env.last_event.frame, env.last_event.metadata)]
        if not args.human_traj:
            while not done:
                # break if max_steps reached
                if t >= args.max_steps:
                    break

                # extract visual features
                curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                feat['frames'] = resnet.featurize([curr_image], batch=1).unsqueeze(0)

                # forward model
                m_out = model.step(feat, t)
                m_pred = model.extract_preds(m_out, [traj_data], feat, clean_special_tokens=False)
                m_pred = list(m_pred.values())[0]


                # check if <<stop>> was predicted
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
                print(f"Step: {t} ", end="\r")
                print('word_action: ', word_action)
                t_success, error, end_inf_state = env.take_action(word_action, num_action, args.rand_agent) # t_success: True, False, or None, or "stop"
                end_inf_state_lst.append((env.last_event.frame, env.last_event.metadata))
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
        else: #human traj
            gt_traj_name = traj_data['root'].rsplit('/', 1)[1]
            gt_traj, lang, scene = cls.get_gt_traj(gt_traj_name) #getting raw traj to see the global coords rather than the tokenized deltas
            for gt_action in gt_traj[1:]:
                t_success, error, end_inf_state = env.take_human_action(gt_action)

                arm_data = end_inf_state["arm"]["joints"][3]['position']
                global_coord_agent = end_inf_state['agent']['position']
                yaw_agent =  end_inf_state['agent']['rotation']['y']
                body_data = [global_coord_agent['x'], global_coord_agent['y'], global_coord_agent['z'], yaw_agent]

                end_inf_state_lst.append(end_inf_state)

        env.i = 0
        
        results = cls.calc_metrics(goal_instr, traj_data['scene'], args.run_save_name, end_inf_state_lst)

    @classmethod
    def get_gt_word_num_actions(cls, action_dict):
        alow = action_dict
        word_action = None
        num_action = None
        if alow['mode'] == 0:
            word_action = 'stop'
            num_action = [0]
        elif alow['mode'] == 1: #base
            if alow['base_action'] == 0:
                word_action = 'NoOp' #made by me
            if alow['base_action'] == 1:
                word_action = 'MoveAhead'
            elif alow['base_action'] == 2:
                word_action = 'MoveBack'
            elif alow['base_action'] == 3:
                word_action = 'MoveRight'
            elif alow['base_action'] == 4:
                word_action = 'MoveLeft'
            num_action = alow['base_action'] #index, not actual val like others, not used down the line
        elif alow['mode'] == 2: #rotate
            word_action = "RotateAgent"
            num_action = alow['state_rot'][0]
        elif alow['mode'] == 3:
            word_action = 'MoveArmBase'
            num_action = alow['state_ee']
        elif alow['mode'] == 4: #end-effector
            word_action = 'MoveArm'
            num_action = alow['state_ee']
        elif alow['mode'] == 5: #grasp/drop
            if alow['grasp_drop'] == 0:
                word_action = 'NoOp' #made by me
            elif alow['grasp_drop'] == 1:
                word_action = 'PickupObject'
            elif alow['grasp_drop'] == 2:
                word_action = 'ReleaseObject'
            num_action = alow['grasp_drop']  #index, not actual val like others, not used down the line
        elif alow['mode'] == 6: #look
            if alow['up_down'] == 0:
                word_action = 'NoOp'
            if alow['up_down'] == 1:
                word_action = 'LookUp'
            elif alow['up_down'] == 2:
                word_action = 'LookDown'
            num_action = alow['up_down']  #index, not actual val like others, not used down the line

        return word_action, num_action

    @classmethod
    def get_gt_traj(cls, gt_traj_name):
        hdf5_file_path = '../../../dataset/sim_dataset.hdf5'
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
                        state_body = steps[0]['state_body'][:3]
                        body_yaw = steps[0]['state_body'][-1]
                        state_ee = steps[0]['state_ee'][:3]
                        
                        state_dict = {'action': action, 'state_body': state_body, 'body_yaw': body_yaw, 'state_ee': state_ee}
                        traj_action_lst.append(state_dict)
        
        return traj_action_lst, nl_command, scene

    @classmethod
    def calc_metrics(cls, task_name, scene_name, split_name, end_inf_state_lst: List[Tuple[np.ndarray, dict]]):
        """
        
        Temporary wrapper method that calls Yichen's method that gets all the metric results. 
        Made it a wrapper so it's more isolated and easier for Yichen to look at. 
        Later, I might remove this wrapper and call his method directly in my main inference code
        end_inf_state a list/rollout the end/last return of each step from the controller from inference episode that has all robot and env info.
        
        """

        results = []
        for img, metadata in end_inf_state_lst:
            action = metadata['lastAction']
            results.append({
                'task': task_name,
                'scene': scene_name,
                'img': img,
                'xyz_body': metadata['agent']['position'],
                'xyz_body_delta': None,
                'yaw_body': metadata['agent']['rotation']['y'],
                'yaw_body_delta': None,
                'pitch_body': None,
                'xyz_ee': metadata['arm']['joints'][3]['position'],
                'xyz_ee_delta': metadata['arm']['joints'][3]['position'],
                'pickup_dropoff': action in ['PickupObject', 'ReleaseObject'],
                'holding_obj': metadata['arm']['heldObjects'],
                'control_mode': None,
                'action': action,
                'terminate': None,
                'step': None,
                'timeout': None,
                'error': metadata['errorMessage']
            })

        final_state = end_inf_state_lst[-1]

        time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = "results/split_{}/traj_{}.pkl"
        os.makedirs("results/split_{}".format(split_name), exist_ok=True)
        with open(filename.format(split_name, time), "wb") as f:
            pickle.dump({
                "trajectory_data": results,
                "final_state": final_state
            }, f)


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