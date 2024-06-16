import cv2
import copy
import numpy as np
from collections import Counter, OrderedDict
import ai2thor
from ai2thor.controller import Controller
from json import load
from os import path
from random import choice, randint
from time import sleep

# DEFAULT_RENDER_SETTINGS = {'renderImage': True,
#                            'renderDepthImage': True,
#                            'renderClassImage': False,
#                            'renderObjectImage': False,
#                            }

class ThorEnv():
    def __init__(self, pp_data_path):

        self.controller = None
        self.last_event = None
        self.i = 0
        self.bins_path = path.join(pp_data_path, 'bins.json')
        with open(self.bins_path, 'r') as f:
            self.bins = load(f)

        print("ThorEnv started.")

    def reset(self, scene_name):
        '''
        reset scene / start scene
        '''
        print("Resetting/starting ThorEnv")
        self.controller = Controller(
            agentMode="arm",
            massThreshold=None,
            scene=scene_name,
            visibilityDistance=1.5,
            gridSize=0.25,
            snapToGrid= False,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=300,
            height=300,
            # width= 1280,
            # height= 720,
            fieldOfView=60
        )
        # self.controller.step(action="Initialize")
        self.last_event = self.controller.last_event
        return self.last_event


    def set_task(self, traj, args, reward_type='sparse', max_episode_length=2000):
        '''
        set the current task type (one of 7 tasks)
        '''
        task_type = traj['task_type']
        self.task = get_task(task_type, traj, self, args, reward_type=reward_type, max_episode_length=max_episode_length)

    def step(self, action, smooth_nav=False):
        '''
        overrides ai2thor.controller.Controller.step() for smooth navigation and goal_condition updates
        '''
        if 'action' in action:
            if smooth_nav:
                if "MoveAhead" in action['action']:
                    self.smooth_move_ahead(action)
                elif "Rotate" in action['action']:
                    self.smooth_rotate(action)
                elif "Look" in action['action']:
                    self.smooth_look(action)
                else:
                    super().step(action)
            else:
                if "LookUp" in action['action']:
                    self.look_angle(-constants.AGENT_HORIZON_ADJ)
                elif "LookDown" in action['action']:
                    self.look_angle(constants.AGENT_HORIZON_ADJ)
                else:
                    super().step(action)
        else:
            super().step(action)

        event = self.update_states(action)
        self.check_post_conditions(action)
        return event


    def get_goal_satisfied(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_satisfied(self.last_event)

    def get_goal_conditions_met(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_conditions_met(self.last_event)

    # def get_subgoal_idx(self):
    #     if self.task is None:
    #         raise Exception("WARNING: no task setup for subgoal_idx")
    #     else:
    #         return self.task.get_subgoal_idx()

    def noop(self):
        '''
        do nothing
        '''
        super().step(dict(action='Pass'))

    def smooth_move_ahead(self, action, render_settings=None):
        '''
        smoother MoveAhead
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        smoothing_factor = constants.RECORD_SMOOTHING_FACTOR
        new_action = copy.deepcopy(action)
        new_action['moveMagnitude'] = constants.AGENT_STEP_SIZE / smoothing_factor

        new_action['renderImage'] = render_settings['renderImage']
        new_action['renderClassImage'] = render_settings['renderClassImage']
        new_action['renderObjectImage'] = render_settings['renderObjectImage']
        new_action['renderDepthImage'] = render_settings['renderDepthImage']

        events = []
        for xx in range(smoothing_factor - 1):
            event = super().step(new_action)
            if event.metadata['lastActionSuccess']:
                events.append(event)

        event = super().step(new_action)
        if event.metadata['lastActionSuccess']:
            events.append(event)
        return events

    def smooth_rotate(self, action, render_settings=None):
        '''
        smoother RotateLeft and RotateRight
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        if action['action'] == 'RotateLeft':
            end_rotation = (start_rotation - 90)
        else:
            end_rotation = (start_rotation + 90)

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def smooth_look(self, action, render_settings=None):
        '''
        smoother LookUp and LookDown
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + constants.AGENT_HORIZON_ADJ * (1 - 2 * int(action['action'] == 'LookUp'))
        position = event.metadata['agent']['position']

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def rotate_angle(self, angle, render_settings=None):
        '''
        rotate at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        end_rotation = start_rotation + angle

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': np.round(end_rotation, 3),
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': horizon,
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(teleport_action)
        return event

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False):
        # TODO: parametrized navigation commands

        if "RotateLeft" in action:
            action = dict(action="RotateLeft",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "RotateRight" in action:
            action = dict(action="RotateRight",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "MoveAhead" in action:
            action = dict(action="MoveAhead",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookUp" in action:
            action = dict(action="LookUp",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookDown" in action:
            action = dict(action="LookDown",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "OpenObject" in action:
            action = dict(action="OpenObject",
                          objectId=object_id,
                          moveMagnitude=1.0)
            event = self.step(action)
        elif "CloseObject" in action:
            action = dict(action="CloseObject",
                          objectId=object_id,
                          forceAction=True)
            event = self.step(action)
        elif "PickupObject" in action:
            action = dict(action="PickupObject",
                          objectId=object_id)
            event = self.step(action)
        elif "PutObject" in action:
            inventory_object_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
            action = dict(action="PutObject",
                          objectId=object_id,
                          forceAction=True,
                          placeStationary=True)
            event = self.step(action)
        elif "ToggleObjectOn" in action:
            action = dict(action="ToggleObjectOn",
                          objectId=object_id)
            event = self.step(action)

        elif "ToggleObjectOff" in action:
            action = dict(action="ToggleObjectOff",
                          objectId=object_id)
            event = self.step(action)
        elif "SliceObject" in action:
            # check if agent is holding knife in hand
            inventory_objects = self.last_event.metadata['inventoryObjects']
            if len(inventory_objects) == 0 or 'Knife' not in inventory_objects[0]['objectType']:
                raise Exception("Agent should be holding a knife before slicing.")

            action = dict(action="SliceObject",
                          objectId=object_id)
            event = self.step(action)
        else:
            raise Exception("Invalid action. Conversion to THOR API failed! (action='" + str(action) + "')")

        return event, action
    def get_first_event(self):
        return self.last_event.metadata
    def init(self):
        fixedDeltaTime = 0.02
        incr = 0.025
        self.controller.step(action="SetHandSphereRadius", radius=0.1)
        self.controller.step(action="MoveArmBase", y=self.i,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
        self.i += incr

    def take_human_action(self, state_action):
        incr = 0.025
        x = 0
        y = 0
        z = 0
        fixedDeltaTime = 0.02
        move = 0.2
        a = None
        word_action = state_action['action']
        print(f'word_action: {word_action}')
        if word_action in ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft']:
            # global_coord_agent = self.last_event.metadata['agent']['position']
            # prev_state_body = [global_coord_agent['x'], global_coord_agent['y'], global_coord_agent['z']]
            # diff = np.array(state_action['state_body']) - np.array(prev_state_body)
            a = dict(action="Teleport", position=dict(x=state_action['state_body'][0], y=state_action['state_body'][1], z=state_action['state_body'][2]))

            # a = dict(action="Teleport", position=dict(x=diff[0], y=diff[1], z=diff[2]))

            # if word_action == "MoveAhead":
            #     a = dict(action="MoveAgent", ahead=move, right=0, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
            # elif word_action == "MoveBack":
            #     a = dict(action="MoveAgent", ahead=-move, right=0, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
            # elif word_action == "MoveRight":
            #     a = dict(action="MoveAgent", ahead=0, right=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
            # elif word_action == "MoveLeft":
            #     a = dict(action="MoveAgent", ahead=0, right=-move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)

            # a = dict(action=word_action, moveMagnitude=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)

        elif word_action in ['PickupObject','ReleaseObject', 'LookUp', 'LookDown']:
            a = dict(action = word_action)
        elif word_action in ['RotateAgent']:
            diff = state_action['body_yaw'] - self.last_event.metadata['agent']['rotation']['y']
            print('rot diff: ', diff)
            a = dict(action="Teleport", rotation=dict(x=0, y=diff, z=0))
            # a = dict(action=word_action, degrees=diff, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
        elif word_action in ['MoveArmBase']:
            prev_ee_y = self.last_event.metadata["arm"]["joints"][3]['position']['y']
            curr_ee_y = state_action['state_ee'][1]
            diff = curr_ee_y - prev_ee_y
            # print(f'prev_ee_y: ', prev_ee_y)
            # print(f'curr_ee_y: ', curr_ee_y)
            # print(f'diff: {diff}')
            if diff > 0:
                self.i += incr
            elif diff < 0:
                self.i -= incr
            a = dict(action="MoveArmBase",y=self.i,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
        elif word_action in ['MoveArm']:
            a = dict(action='MoveArm',position=dict(x=state_action['state_ee'][0], y=state_action['state_ee'][1], z=state_action['state_ee'][2]),coordinateSpace="world",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)

        sleep(0.35) #for debugging/movement analysis
        # if a['action'] == 'Teleport' and word_action in ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft']:
            # breakpoint()
        event = self.controller.step(a)
        success = event.metadata['lastActionSuccess']
        error = event.metadata['errorMessage']
        self.last_event = event
        return success, error, self.last_event.metadata


    def take_action(self, word_action, num_action, rand_agent=False):
        # i = 0
        incr = 0.025
        x = 0
        y = 0
        z = 0
        fixedDeltaTime = 0.02
        move = 0.2
        a = None

        if rand_agent:
            all_word_actions = ['PickupObject','ReleaseObject', 'LookUp', 'LookDown', 'MoveArm', 'MoveArmBase', 'RotateAgent', 'MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft', 'stop']
            rand_word_action = choice(all_word_actions)
            if rand_word_action in ["stop"]:
                return "stop", None
            elif rand_word_action in ['PickupObject','ReleaseObject', 'LookUp', 'LookDown']:
                a = dict(action = rand_word_action)
            elif rand_word_action in ['MoveArm', 'MoveArmBase']:
                global_coord_ee = self.last_event.metadata["arm"]["joints"][3]['position']
                curr_x, curr_y, curr_z = global_coord_ee['x'], global_coord_ee['y'], global_coord_ee['z']
                rand_x_indx, rand_y_indx, rand_z_indx = randint(1, 256), randint(1, 256), randint(1, 256) # starts at 1 to skip NoOp
                x_del, y_del, z_del = self.bins["4"][rand_x_indx], self.bins["5"][rand_y_indx], self.bins["6"][rand_z_indx]
                new_x, new_y, new_z = curr_x + x_del, curr_y + y_del, curr_z + z_del
                a = dict(action='MoveArm', position=dict(x=new_x, y=new_y, z=new_z),coordinateSpace="world",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
            elif rand_word_action in ['RotateAgent']:
                rand_yaw_indx = randint(1, 256)
                new_yaw = self.bins["3"][rand_yaw_indx]
                a = dict(action=rand_word_action, degrees=new_yaw, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
            else: # move base
                a = dict(action=rand_word_action, moveMagnitude=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)

        else:
            if word_action in ['NoOp']:
                print(f"Word Action: NoOP", end="\r") # for debugging
                if self.last_event == None:
                    return None, None, None
                return None, None, self.last_event.metadata
            elif word_action in ['PickupObject','ReleaseObject', 'LookUp', 'LookDown']:
                a = dict(action = word_action)
            elif word_action in ['MoveArmBase']:
                ee_y_delta = num_action[1]
                if ee_y_delta > 0:
                    self.i += incr
                elif ee_y_delta < 0:
                    self.i -= incr
                a = dict(action="MoveArmBase",y=self.i,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
            elif word_action in ['MoveArm']:
                global_coord_ee = self.last_event.metadata["arm"]["joints"][3]['position']
                curr_x, curr_y, curr_z = global_coord_ee['x'], global_coord_ee['y'], global_coord_ee['z']
                x_del, y_del, z_del = self.bins["4"][num_action[0]], self.bins["5"][num_action[1]], self.bins["6"][num_action[2]]
                if x_del == -1000 or y_del == -1000 or z_del == -1000: # if any of them are NoOp then skip all. Can do it another way where only skip the specific axis
                    print(f"Word Action: NoOP", end="\r") # for debugging
                    if self.last_event == None:
                        return None, None, None
                    return None, None, self.last_event.metadata
                new_x, new_y, new_z = curr_x + x_del, curr_y + y_del, curr_z + z_del
                a = dict(action='MoveArm',position=dict(x=new_x, y=new_y, z=new_z),coordinateSpace="world",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
            elif word_action in ['RotateAgent']:
                yaw_del = num_action if type(num_action) == int else num_action.item()
                new_yaw = self.bins["3"][yaw_del]
                if new_yaw == -1000: #make it variable later
                    print(f"Word Action: NoOP", end="\r") # for debugging
                    if self.last_event == None:
                        return None, None, None
                    return None, None, self.last_event.metadata
            
                # a = dict(action=word_action, degrees=new_yaw,returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
                a = dict(action="Teleport", rotation=dict(x=0, y=new_yaw, z=0))
            else: # move base
                # a = dict(action=word_action, moveMagnitude=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
                if word_action == "MoveAhead":
                    a = dict(action="MoveAgent", ahead=move, right=0, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
                elif word_action == "MoveBack":
                    a = dict(action="MoveAgent", ahead=-move, right=0, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
                elif word_action == "MoveRight":
                    a = dict(action="MoveAgent", ahead=0, right=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
                elif word_action == "MoveLeft":
                    a = dict(action="MoveAgent", ahead=0, right=-move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)

        sleep(1) #for debugging/movement analysis
        event = self.controller.step(a)
        success = event.metadata['lastActionSuccess']
        error = event.metadata['errorMessage']
        self.last_event = event
        #for debugging/movement analysis
        sleep(0.5)
        if rand_agent:
            print(f"Random Word Action: {rand_word_action} ", end="\r")
        # else:
            # print(f"Word Action: {word_action} ", end="\r")
            # print(f"Num Action: {num_action} ", end="\r")

        return success, error, self.last_event.metadata