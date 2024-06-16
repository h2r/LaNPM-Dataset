from ai2thor.controller import Controller
from ai2thor.server import MultiAgentEvent
from metrics.base_metric import TrajData
from .base_metric import Metric

TASK_SUCC_MAPPING = {

}


def extract_task_succ(first_gt_metadata, last_gt_metadata, cmd):
    diff_list = []
    sorted_first = sorted(first_gt_metadata['objects'], key=lambda item: item["name"])
    sorted_last  = sorted(last_gt_metadata['objects'], key=lambda item: item["name"])
    for first, last in zip(sorted_first, sorted_last):
        if first['name'] != last['name']:
            raise ValueError("Item type got changed during execution")
        if first['position'] != last['position']:
            diff_list.append(("position", last['name'], last['position']))
        if first['receptableObjectIds'] != last['receptableObjectIds']:
            diff_list.append(("receptables", last['name'], last['receptableObjectIds']))
    return diff_list
    
if __name__ == "__main__":
    with open("metrics/1.json", 'r') as f:
        import json
        a = json.load(f)
    extract_task_succ(a, a, "123")

class TaskSuccMetric(Metric):
    def __init__(self, name="success_rate"):
        self.name = name
    
    def get_score(
            self, 
            scene_name: str, 
            traj_model: TrajData, 
            traj_gt: TrajData, 
            final_state: MultiAgentEvent, 
            task_cmd: str,
            first_gt_metadata: dict=None,
            last_gt_metadata: dict=None
        ) -> float:
        final_state.last_event
        # mode 1: "near" for the target
        if "near" in task_cmd.split("and")[-1]:
            # check location changes
            pass
            
        # mode 2: "on" for the target
        else:
            pass

        # compare first and last metadata
