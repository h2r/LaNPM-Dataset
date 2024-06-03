from typing import List, Mapping
from dataclasses import dataclass
import numpy as np

from ai2thor.controller import Controller


@dataclass
class TrajData:
    """
    Data for a single trajectory, contain a sequence of images, body position, body yaw, and end effector position,
    each represented as a numpy array of size n x [...] where n is the length of the trajs.
    """
    img: np.ndarray # nxwxhx3
    xyz_body: np.ndarray # nx3, global position of the body
    yaw_body: np.ndarray # nx1
    xyz_ee: np.ndarray # nx3, global position of the end effector


class Metric:
    def __init__(self, name):
        self.name = name

    def get_score(
            self, 
            scene_name: str, 
            traj: TrajData, 
            final_state: Controller, 
            task_cmd: str
        ) -> float:
        raise NotImplementedError("Subclasses should implement this!")
