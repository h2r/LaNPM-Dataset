from ai2thor.controller import Controller
from metrics.base_metric import TrajData
from .base_metric import Metric

TASK_SUCC_MAPPING = {

}



class TaskSuccMetric(Metric):
    def get_score(
            self, 
            scene_name: str, 
            traj_model: TrajData, 
            traj_gt: TrajData, 
            final_state: Controller, 
            task_cmd: str,
            first_gt_metadata: dict=None,
            last_gt_metadata: dict=None
        ) -> float:

        # compare first and last metadata
        pass