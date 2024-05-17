import cv2
import copy
import gen.constants as constants
import numpy as np
from collections import Counter, OrderedDict
from env.tasks import get_task
import ai2thor
from ai2thor.controller import Controller
import gen.utils.image_util as image_util
from gen.utils import game_util
from gen.utils.game_util import get_objects_of_type, get_obj_of_type_closest_to_obj


# DEFAULT_RENDER_SETTINGS = {'renderImage': True,
#                            'renderDepthImage': True,
#                            'renderClassImage': False,
#                            'renderObjectImage': False,
#                            }

class ThorEnv():
    def __init__(self):

        self.controller = None
        self.last_event = None

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
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=300,
            height=300,
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

    def take_action(self, word_action, num_action):
        i = 0
        incr = 0.025
        x = 0
        y = 0
        z = 0
        fixedDeltaTime = 0.02
        move = 0.2

        a = None
        if word_action in ['PickupObject','ReleaseObject', 'LookUp', 'LookDown']:
            a = dict(action = word_action)
        elif word_action in ['MoveArm']
            global_coord_ee = self.last_event.metadata["arm"]["joints"][3]['position']
            curr_x, curr_y, curr_z = global_coord_ee['x'], global_coord_ee['y'], global_coord_ee['z']
            x_del, y_del, z_del = num_action
            new_x, new_y, new_z = curr_x + x_del, curr_y + y_del, curr_z + z_del
            a = dict(action=word_action,position=dict(x=new_x, y=new_y, z=new_z),coordinateSpace="global",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)


        event = self.controller.step(a)