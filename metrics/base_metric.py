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


class DeltaDist(Metric):
    def __init__(self, name="delta_loc"):
        self.name = name
    
    def get_score(self, scene_name: str, traj_model: TrajData, traj_gt: TrajData, final_state: Controller, task_cmd: str):
        diff_xyz_body = traj_model.xyz_body[1:] - traj_model.xyz_body[:-1]
        diff_yaw_body = traj_model.yaw_body[1:] - traj_model.yaw_body[:-1]
        diff_xyz_ee = traj_model.xyz_ee[1:] - traj_model.xyz_ee[:-1]

        euclidean_xyz_body = np.sqrt(np.sum(diff_xyz_body ** 2, axis=1))
        euclidean_yaw_body = np.abs(diff_yaw_body)
        euclidean_xyz_ee = np.sqrt(np.sum(diff_xyz_ee ** 2, axis=1))

        return {
            "xyz_body/mean": euclidean_xyz_body.mean(),
            "xyz_body/std": euclidean_xyz_body.std(),
            "yaw_body/mean": euclidean_yaw_body.mean(),
            "yaw_body/std": euclidean_yaw_body.std(),
            "xyz_ee/mean": euclidean_xyz_ee.mean(),
            "xyz_ee/std": euclidean_xyz_ee.std()
        }



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
        body_coordinates = np.array(traj_model.xyz_body)[:,[0,2,1]]


        hull = ConvexHull(body_coordinates[:,:2])


        # Get the vertices of the convex hull
        hull_points = traj_model.xyz_body[hull.vertices]

        # Create a polygon from the hull points
        polygon = Polygon(hull_points)

        # Compute the area of the polygon
        traversed_area = polygon.area

        #compute area coverage per unit of trajectory
        area_per_step = traversed_area / len(traj_model.xyz_body)

        #NOTE: For debugging, can remove later
        #body_coordinates1 = [[4.0, 0.9009992480278015, -4.0], [4.0, 0.9009992480278015, -4.0], [3.819993734359741, 0.9009992480278015, -3.4000020027160645], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [3.819993734359741, 0.9009992480278015, -3.4000020027160645], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [3.819993734359741, 0.9009992480278015, -3.4000020027160645], [3.819993734359741, 0.9009992480278015, -3.4000020027160645], [3.6199936866760254, 0.9009992480278015, -3.4000041484832764], [3.6199958324432373, 0.9009992480278015, -3.600004196166992], [3.6199958324432373, 0.9009992480278015, -3.600004196166992], [3.6199958324432373, 0.9009992480278015, -3.600004196166992], [3.6199958324432373, 0.9009992480278015, -3.600004196166992], [3.6199958324432373, 0.9009992480278015, -3.600004196166992], [3.999997854232788, 0.9009992480278015, -3.799999952316284], [3.6199958324432373, 0.9009992480278015, -3.600004196166992], [3.6199958324432373, 0.9009992480278015, -3.600004196166992], [3.6199936866760254, 0.9009992480278015, -3.4000041484832764], [3.619993209838867, 0.9009992480278015, -3.360004186630249], [3.619993209838867, 0.9009992480278015, -3.360004186630249], [3.619993209838867, 0.9009992480278015, -3.360004186630249], [3.619993209838867, 0.9009992480278015, -3.360004186630249], [3.619993209838867, 0.9009992480278015, -3.360004186630249], [3.619993209838867, 0.9009992480278015, -3.360004186630249], [3.619993209838867, 0.9009992480278015, -3.360004186630249], [3.999995708465576, 0.9009992480278015, -3.5999999046325684], [3.619993209838867, 0.9009992480278015, -3.360004186630249], [3.619993209838867, 0.9009992480278015, -3.360004186630249], [3.619993209838867, 0.9009992480278015, -3.360004186630249], [3.619993209838867, 0.9009992480278015, -3.360004186630249], [3.816955089569092, 0.9009992480278015, -3.3947317600250244], [3.8760437965393066, 0.9009992480278015, -3.4051501750946045], [3.910771369934082, 0.9009992480278015, -3.20818829536438], [3.9177169799804688, 0.9009992480278015, -3.1687958240509033], [4.114678859710693, 0.9009992480278015, -3.2035233974456787], [4.114678859710693, 0.9009992480278015, -3.2035233974456787], [3.9999935626983643, 0.9009992480278015, -3.3999998569488525], [4.31164026260376, 0.9009992480278015, -3.1687917709350586], [4.508601665496826, 0.9009992480278015, -3.1340601444244385], [4.705563068389893, 0.9009992480278015, -3.0993285179138184], [4.902524471282959, 0.9009992480278015, -3.0645968914031982], [4.937256336212158, 0.9009992480278015, -3.2615580558776855], [5.134217739105225, 0.9009992480278015, -3.2268264293670654], [5.331179141998291, 0.9009992480278015, -3.1920948028564453], [5.331179141998291, 0.9009992480278015, -3.1920948028564453], [5.331179141998291, 0.9009992480278015, -3.1920948028564453], [5.483153820037842, 0.9009992480278015, -3.3221089839935303], [3.819993734359741, 0.9009992480278015, -3.4000020027160645], [5.635128498077393, 0.9009992480278015, -3.4521231651306152], [5.787103176116943, 0.9009992480278015, -3.5821373462677], [5.917117118835449, 0.9009992480278015, -3.4301626682281494], [5.917117118835449, 0.9009992480278015, -3.4301626682281494], [5.917117118835449, 0.9009992480278015, -3.4301626682281494], [5.917117118835449, 0.9009992480278015, -3.4301626682281494], [6.1057000160217285, 0.9009992480278015, -3.363555669784546], [6.294282913208008, 0.9009992480278015, -3.2969486713409424], [6.482865810394287, 0.9009992480278015, -3.230341672897339], [6.671448707580566, 0.9009992480278015, -3.1637346744537354], [3.819993734359741, 0.9009992480278015, -3.4000020027160645], [6.604841709136963, 0.9009992480278015, -2.975151777267456], [6.538234710693359, 0.9009992480278015, -2.7865688800811768], [6.471627712249756, 0.9009992480278015, -2.5979859828948975], [6.660210609436035, 0.9009992480278015, -2.531378984451294], [6.8487935066223145, 0.9009992480278015, -2.4647719860076904], [6.8487935066223145, 0.9009992480278015, -2.4647719860076904], [6.8487935066223145, 0.9009992480278015, -2.4647719860076904], [6.886387825012207, 0.9009992480278015, -2.2683372497558594], [6.886387825012207, 0.9009992480278015, -2.2683372497558594], [6.965856552124023, 0.9009992480278015, -2.084803342819214], [3.819993734359741, 0.9009992480278015, -3.4000020027160645], [7.04532527923584, 0.9009992480278015, -1.9012694358825684], [7.124794006347656, 0.9009992480278015, -1.7177355289459229], [7.124794006347656, 0.9009992480278015, -1.7177355289459229], [7.124794006347656, 0.9009992480278015, -1.7177355289459229], [7.303644180297852, 0.9009992480278015, -1.6282219886779785], [7.482494354248047, 0.9009992480278015, -1.5387084484100342], [7.482494354248047, 0.9009992480278015, -1.5387084484100342], [7.681173801422119, 0.9009992480278015, -1.515763521194458], [7.879853248596191, 0.9009992480278015, -1.4928185939788818], [8.078533172607422, 0.9009992480278015, -1.4698736667633057], [3.819993734359741, 0.9009992480278015, -3.4000020027160645], [8.277213096618652, 0.9009992480278015, -1.4469287395477295], [8.277213096618652, 0.9009992480278015, -1.4469287395477295], [8.277213096618652, 0.9009992480278015, -1.4469287395477295], [8.277213096618652, 0.9009992480278015, -1.4469287395477295], [8.441932678222656, 0.9009992480278015, -1.5603634119033813], [8.60665225982666, 0.9009992480278015, -1.6737980842590332], [8.60665225982666, 0.9009992480278015, -1.6737980842590332], [8.60665225982666, 0.9009992480278015, -1.6737980842590332], [8.60665225982666, 0.9009992480278015, -1.6737980842590332], [8.60665225982666, 0.9009992480278015, -1.6737980842590332], [3.819993734359741, 0.9009992480278015, -3.4000020027160645], [8.60665225982666, 0.9009992480278015, -1.6737980842590332], [8.60665225982666, 0.9009992480278015, -1.6737980842590332], [8.60665225982666, 0.9009992480278015, -1.6737980842590332], [8.771371841430664, 0.9009992480278015, -1.787232756614685], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337], [8.936091423034668, 0.9009992480278015, -1.900667428970337]]
        # body_coordinates = [[4.0, 0.9009992480278015, -4.0], [4.0, 0.9009992480278015, -4.0], [4.000009059906006, 0.9009992480278015, -4.679999351501465], [4.276866912841797, 0.9009992480278015, -3.2882847785949707], [4.473828315734863, 0.9009992480278015, -3.2535529136657715], [4.67078971862793, 0.9009992480278015, -3.2188210487365723], [4.867751121520996, 0.9009992480278015, -3.184089183807373], [5.0647125244140625, 0.9009992480278015, -3.149357318878174], [5.261673927307129, 0.9009992480278015, -3.1146254539489746], [5.296405792236328, 0.9009992480278015, -3.311586618423462], [5.4933671951293945, 0.9009992480278015, -3.2768547534942627], [5.690328598022461, 0.9009992480278015, -3.2421228885650635], [5.887290000915527, 0.9009992480278015, -3.2073910236358643], [3.9015278816223145, 0.9009992480278015, -4.697365760803223], [6.084251403808594, 0.9009992480278015, -3.172659158706665], [6.084251403808594, 0.9009992480278015, -3.172659158706665], [6.257455348968506, 0.9009992480278015, -3.0726571083068848], [6.257455348968506, 0.9009992480278015, -3.0726571083068848], [6.257455348968506, 0.9009992480278015, -3.0726571083068848], [6.257455348968506, 0.9009992480278015, -3.0726571083068848], [6.257452964782715, 0.9009992480278015, -2.872657060623169], [6.257450580596924, 0.9009992480278015, -2.672657012939453], [6.4574503898620605, 0.9009992480278015, -2.672654867172241], [6.657450199127197, 0.9009992480278015, -2.6726527214050293], [3.8667960166931152, 0.9009992480278015, -4.500404357910156], [6.857450008392334, 0.9009992480278015, -2.6726505756378174], [6.857447624206543, 0.9009992480278015, -2.4726505279541016], [6.857445240020752, 0.9009992480278015, -2.2726504802703857], [6.857442855834961, 0.9009992480278015, -2.07265043258667], [6.85744047164917, 0.9009992480278015, -1.872650384902954], [6.85744047164917, 0.9009992480278015, -1.872650384902954], [6.85744047164917, 0.9009992480278015, -1.872650384902954], [6.985996246337891, 0.9009992480278015, -1.71943998336792], [7.114552021026611, 0.9009992480278015, -1.5662295818328857], [7.114552021026611, 0.9009992480278015, -1.5662295818328857], [3.669834852218628, 0.9009992480278015, -4.5351362228393555], [7.287755966186523, 0.9009992480278015, -1.466227650642395], [7.4609599113464355, 0.9009992480278015, -1.3662257194519043], [7.634163856506348, 0.9009992480278015, -1.2662237882614136], [7.634163856506348, 0.9009992480278015, -1.2662237882614136], [7.831125259399414, 0.9009992480278015, -1.2314919233322144], [7.9296064376831055, 0.9009992480278015, -1.2141261100769043], [7.964338302612305, 0.9009992480278015, -1.4110872745513916], [8.161299705505371, 0.9009992480278015, -1.3763554096221924], [8.358261108398438, 0.9009992480278015, -1.3416235446929932], [8.358261108398438, 0.9009992480278015, -1.3416235446929932], [3.4728736877441406, 0.9009992480278015, -4.569868087768555], [8.358261108398438, 0.9009992480278015, -1.3416235446929932], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [3.2759125232696533, 0.9009992480278015, -4.604599952697754], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.51147174835205, 0.9009992480278015, -1.4701793193817139], [8.664682388305664, 0.9009992480278015, -1.5987350940704346], [8.664682388305664, 0.9009992480278015, -1.5987350940704346], [8.664682388305664, 0.9009992480278015, -1.5987350940704346], [3.241180658340454, 0.9009992480278015, -4.4076385498046875], [8.664682388305664, 0.9009992480278015, -1.5987350940704346], [8.664682388305664, 0.9009992480278015, -1.5987350940704346], [8.664682388305664, 0.9009992480278015, -1.5987350940704346], [8.664682388305664, 0.9009992480278015, -1.5987350940704346], [8.664682388305664, 0.9009992480278015, -1.5987350940704346], [8.536126136779785, 0.9009992480278015, -1.7519454956054688], [8.471848487854004, 0.9009992480278015, -1.8285505771636963], [8.502490043640137, 0.9009992480278015, -1.8542616367340088], [3.044219493865967, 0.9009992480278015, -4.442370414733887], [2.8472583293914795, 0.9009992480278015, -4.477102279663086], [2.650297164916992, 0.9009992480278015, -4.511834144592285], [4.000002384185791, 0.9009992480278015, -4.199999809265137], [2.453336000442505, 0.9009992480278015, -4.546566009521484], [2.2563748359680176, 0.9009992480278015, -4.581297874450684], [2.2216429710388184, 0.9009992480278015, -4.384336471557617], [2.186911106109619, 0.9009992480278015, -4.187375068664551], [2.15217924118042, 0.9009992480278015, -3.9904139041900635], [2.1174473762512207, 0.9009992480278015, -3.793452739715576], [2.1174473762512207, 0.9009992480278015, -3.793452739715576], [2.1174473762512207, 0.9009992480278015, -3.793452739715576], [2.1174473762512207, 0.9009992480278015, -3.793452739715576], [1.929508090019226, 0.9009992480278015, -3.725050687789917], [4.000004768371582, 0.9009992480278015, -4.399999618530273], [1.7415688037872314, 0.9009992480278015, -3.656648635864258], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [4.000007152557373, 0.9009992480278015, -4.59999942779541], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [4.000009059906006, 0.9009992480278015, -4.679999351501465], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [4.000009059906006, 0.9009992480278015, -4.679999351501465], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [1.809970736503601, 0.9009992480278015, -3.4687092304229736], [4.000009059906006, 0.9009992480278015, -4.679999351501465], [1.9831769466400146, 0.9009992480278015, -3.568707227706909], [2.1563830375671387, 0.9009992480278015, -3.6687052249908447], [2.056385040283203, 0.9009992480278015, -3.8419113159179688], [1.956386923789978, 0.9009992480278015, -4.015117645263672], [1.856388807296753, 0.9009992480278015, -4.188323974609375], [1.856388807296753, 0.9009992480278015, -4.188323974609375], [1.856388807296753, 0.9009992480278015, -4.188323974609375], [2.0533499717712402, 0.9009992480278015, -4.153592109680176], [2.2503111362457275, 0.9009992480278015, -4.118860244750977], [2.3290958404541016, 0.9009992480278015, -4.10496711730957], [4.000009059906006, 0.9009992480278015, -4.679999351501465], [2.363827705383301, 0.9009992480278015, -4.301928520202637], [2.560788869857788, 0.9009992480278015, -4.2671966552734375], [2.7577500343322754, 0.9009992480278015, -4.232464790344238], [2.9153199195861816, 0.9009992480278015, -4.204678535461426], [2.950051784515381, 0.9009992480278015, -4.401639938354492], [3.147012948989868, 0.9009992480278015, -4.366908073425293], [3.3439741134643555, 0.9009992480278015, -4.332176208496094], [3.5409352779388428, 0.9009992480278015, -4.2974443435668945], [3.73789644241333, 0.9009992480278015, -4.262712478637695], [3.73789644241333, 0.9009992480278015, -4.262712478637695], [4.000009059906006, 0.9009992480278015, -4.679999351501465], [3.73789644241333, 0.9009992480278015, -4.262712478637695], [3.73789644241333, 0.9009992480278015, -4.262712478637695], [3.80629825592041, 0.9009992480278015, -4.07477331161499], [3.8747000694274902, 0.9009992480278015, -3.886834144592285], [3.9431018829345703, 0.9009992480278015, -3.69889497756958], [4.01150369644165, 0.9009992480278015, -3.510955810546875], [4.0799055099487305, 0.9009992480278015, -3.32301664352417], [4.0799055099487305, 0.9009992480278015, -3.32301664352417], [4.0799055099487305, 0.9009992480278015, -3.32301664352417], [4.0799055099487305, 0.9009992480278015, -3.32301664352417]]
        # body_coordinates = np.array(body_coordinates)[:,[0,2,1]]
        # hull = ConvexHull(body_coordinates[:,:2])
        score = {'total_area': traversed_area, 'area_per_step': area_per_step}
        
        return score



class CLIP_SemanticUnderstanding(Metric):

    def __init__(
            self, 
            name='clip_semantic_understanding', 
            bellman_lambda=0.99, 
            ema_interval=10,
            scene_to_cmds={},
        ):

        '''
        1. Bellman like discounted reward equation
        2. Exponential moving average
        3. Task v.s. other task prediction
        '''
        self.name = name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=self.device)
        
        #hyperparams for the CLIP metric
        self.bellman_lambda = bellman_lambda
        self.ema_alpha = 2 / (ema_interval + 1)
        self.ema_interval = ema_interval

        #list of all tasks to be performed in the current scene
        self.sceme_to_cmds = scene_to_cmds


    def get_score(self, scene_name: str, traj_model: TrajData, traj_gt: TrajData, final_state: Controller, task_cmd: str):
        # assert(len(traj_model[traj_model.keys()[0]]) == len(traj_gt[traj_gt.keys()[0]]), "Error: GT and model traj don't have the same length")
        assert 'img' in dir(traj_model) and 'img' in dir(traj_gt), "Error: image key not in model or gt trajectory"
        
        discounted_clip_reward = 0.0
        ema_clip_reward = 0.0

        for i in range(traj_model.img.shape[0]):
            preprocessed_image = self.clip_preprocess(Image.fromarray(traj_model.img[i])).unsqueeze(0).to(self.device)
            preprocessed_text = clip.tokenize([task_cmd]).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(preprocessed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                text_features = self.clip_model.encode_text(preprocessed_text)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                clip_similarity_score = image_features @ text_features.T
            
            discounted_clip_reward += self.bellman_lambda*clip_similarity_score if i < traj_model.img.shape[0] - 1 else clip_similarity_score

            if (i+1) <= self.ema_interval:
                ema_clip_reward += clip_similarity_score / self.ema_interval
            else:
                ema_clip_reward = self.ema_alpha*clip_similarity_score + (1-self.ema_alpha)*ema_clip_reward

        all_tasks_for_scene = self.sceme_to_cmds[scene_name]
        success_index = all_tasks_for_scene.index(task_cmd)

        correct_task_clip_score = 0

        for i in range(traj_model.img.shape[0]):
            preprocessed_image = self.clip_preprocess(Image.fromarray(traj_model.img[i])).unsqueeze(0).to(self.device)
            preprocessed_text = clip.tokenize(all_tasks_for_scene).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(preprocessed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                text_features = self.clip_model.encode_text(preprocessed_text)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                clip_similarity_score = image_features @ text_features.T
            
            if np.argmax(clip_similarity_score.cpu().numpy(), axis=1) == success_index:
                correct_task_clip_score += 1

        correct_task_clip_score = correct_task_clip_score / traj_model.img.shape[0]

        scores = {'ema_clip_reward': ema_clip_reward.cpu().item(), 'discounted_clip_reward': discounted_clip_reward.cpu().item(), 'correct_task_clip_score': correct_task_clip_score}
        return scores
