# LaNPM Dataset Benchmark
As robots that follow natural language become more capable and prevalent, we need a benchmark to holistically develop and evaluate their ability to solve long-horizon mobile manipulation tasks in large, diverse environments. Robots must use visual and language understanding, navigation, and manipulation capabilities to tackle this challenge. Existing datasets do not integrate all these aspects, restricting their efficacy as benchmarks. To address this gap, we present the Language, Navigation, Manipulation, Perception (LaNMP) dataset and demonstrate the benefits of integrating these four capabilities and various modalities. LaNMP comprises 574 trajectories across eight simulated and real-world environments for long-horizon room-to-room pick-and-place tasks specified by natural language. Every trajectory consists of over 20 attributes, including RGB-D images, segmentations, and the poses of the robot body, end-effector, and grasped objects. We fine-tuned and tested two models in simulation and on a physical robot to demonstrate its efficacy in development and evaluation. The models perform suboptimally compared to humans across various metrics, indicating significant room for developing better multimodal mobile manipulation models using our benchmark.

![Sequential timesteps of images from sim and real collected robot trajectories along with the natural language command describing the task.](./media/Trajectories-Figure.png "Sim and real trajectories")

## Dataset Format
More detailed dataset information can be found in the dataset card [DataCard.md](https://github.com/h2r/LaNPM-Dataset/blob/main/DataCard.md#lanmp).

### Sim Dataset
The simulation dataset comes in a single hdf5 file, and has the following hierarchy:
```
sim_dataset.hdf5/
├── data_11:11:28/
│   ├── folder_0
│   ├── folder_1
│   └── folder_2
├── data_11:14:08/
│   ├── folder_0
│   └── ...
└── ...
```

Under each folder, there are three main numpy files: `depth_<num>`, `inst_seg_<num>`, and `rgb_<num>`,
which correspond to the depth image, segmentation image, and rgb image, respectively.

Under the metadata for each folder, there is a dumped json describing other metadata of each time step.
The detailed metadata can be found in the dataset card.

<!-- | key | description | value |
|---|---|---|
| sim_time | Simulation time from game start | 0.1852477639913559 |
| wall-clock_time | Wall clock time at the step | 15:10:47.900 |
| action | Discrete Action to be executed | Initialize |
| state_body | State of the body, in x, y, z, and yaw. | [3.0, 0.9009992480278015, -4.5, 269.9995422363281] |
| state_ee | End Effector state, in x, y, z, roll, pitch, yaw. | [2.5999975204467773, 0.8979992270469666, -4.171003341674805, -1.9440563492718068e-07, -1.2731799533306385, 1.9440386333307377e-07] |
| held_objs | list of held objects | [] |
| held_objs_state | list of state of held objects | {} |
| inst_det2D | Object Detection Data | {"keys": ["Wall_4\|0.98\|1.298\|-2.63", "Wall_3\|5.43\|1.298\|-5.218", "RemoteControl\|+01.15\|+00.48\|-04.24", "AlarmClock\|+01.31\|+00.48\|-04.01", "wall_panel_32_5 (14)\|1.978\|0\|-4.912", "wall_panel_32_5 (12)\|1\|0\|-3.934", "wall_panel_32_5 (27)\|1.977441\|0\|-3.787738", "wall_panel_32_5 (26)\|1\|0\|-3.787738", "wall_panel_32_5 (11)\|1\|0\|-2.956", "wall_panel_32_5 (13)\|1\|0\|-4.912", "SideTable\|+01.21\|+00.00\|-04.25", "RoboTHOR_Ceiling\|0\|0\|0"], "values": [[418, 43, 1139, 220], [315, 0, 417, 113], [728, 715, 760, 719], [785, 687, 853, 719], [0, 0, 393, 719], [514, 196, 816, 719], [1071, 0, 1279, 719], [860, 41, 1071, 719], [816, 196, 859, 719], [392, 42, 514, 719], [591, 711, 785, 719], [389, 0, 1207, 42]]} |
| rgb | RGB Image path | ./rgb_0.npy |
| depth | Depth Image Path | ./depth_0.npy |
| inst_seg | Segmentation Image Path | ./inst_seg_0.npy |
| hand_sphere_radius | Radius of simulated hand sphere | 0.05999999865889549 | -->


### Real Dataset
Similarly, the real dataset also comes in a single hdf5 file, and has the following hierarchy:
```
real_dataset.hdf5/
└── FloorTrajectories/
    ├── data_00/
    │   ├── folder_10/
    │   │   ├── gripper_depth_10
    │   │   ├── gripper_image_10
    │   │   ├── left_fisheye_depth_10
    │   │   ├── left_fisheye_image_10
    │   │   ├── right_fisheye_depth_10
    │   │   ├── right_fisheye_image_10
    │   │   └── metadata
    │   └── folder_11/
    │       ├── gripper_depth_10
    │       ├── gripper_image_10
    │       └── ...
    ├── data_01/
    │   └── folder_10/
    │       └── ...
    └── ...
```
Note that the right fisheye is located on the right side of the robot, but points towards the left side.
So the right fisheye produces the left half of the image, and the left one produces the right half.

The images have the following sizes:
| key | shape |
|---|---|
| gripper_depth_10 | (480, 640) |
| gripper_image_10 | (480, 640, 3) |
| left_fisheye_depth_10 | (240, 424) |
| left_fisheye_image_10 | (640, 480, 3) |
| right_fisheye_depth_10 | (240, 424) |
| right_fisheye_image_10 | (640, 480, 3) |


The detailed metadata can be found in the dataset card.

<!-- | key | value |
|---|---|
| language_command | Go to the toy kitchen that is to the right when you exit the room, grab the plastic green pepper, go to the kitchen area in the main room, place it on top of the kitchen counter. |
| scene_name |  |
| wall_clock_time | 13:50:11.201 |
| left_fisheye_rgb | left_fisheye_image_0.npy |
| left_fisheye_depth | left_fisheye_depth_0.npy |
| right_fisheye_rgb | right_fisheye_image_0.npy |
| right_fisheye_depth | right_fisheye_depth_0.npy |
| gripper_rgb | gripper_image_0.npy |
| gripper_depth | gripper_depth_0.npy |
| left_fisheye_instance_seg | left_fisheye_image_instance_seg_0.npy |
| right_fisheye_instance_seg | right_fisheye_image_instance_seg_0.npy |
| gripper_fisheye_instance_seg | gripper_image_instance_seg_0.npy |
| body_state | {"x": 1.019768863596449, "y": -0.12653324851702852, "z": 0.038452945167719146} |
| body_quaternion | {"w": 0.07045753575836211, "x": 0.0018112967531622903, "y": 0.001095438062942932, "z": 0.9975125336928763} |
| body_orientation | {"r": 0.0016598882407883365, "p": 0.013164860536916328, "y": 3.000247603448528} |
| body_linear_velocity | {"x": -0.00023874381943789457, "y": 0.0007513785792433702, "z": 0.00019488704919604812} |
| body_angular_velocity | {"x": 0.003917423993358034, "y": 1.1937603762667328e-05, "z": -0.002981354306862609} |
| arm_state_rel_body | {"x": 0.5536243915557861, "y": -5.9951755247311667e-05, "z": 0.2608567476272583} |
| arm_quaternion_rel_body | {"w": 0.9999653697013855, "x": -0.0003896613488905132, "y": 0.008311624638736248, "z": 0.008311624638736248} |
| arm_orientation_rel_body | {"x": -0.000782161177804642, "y": 0.016623309930268487, "z": -0.0003317543202410178} |
| arm_state_global | {"x": 0.4726361013872905, "y": -5.9951755247311667e-05, "z": 0.2608567476272583} |
| arm_quaternion_global | {"w": 0.07061215553290562, "x": -0.006507352005362216, "y": 0.0012926250807351186, "z": 0.9974817843132443} |
| arm_orientation_global | {"x": 0.0016598882407883365, "y": 0.013164860536916328, "z": 3.000247603448528} |
| arm_linear_velocity | {"x": -0.0013432701884599117, "y": 0.003288924836409269, "z": -0.0091181390098158} |
| arm_angular_velocity | {"x": 0.005117543471770197, "y": -0.023086599953784714, "z": -0.008514789140292673} |
| arm_stowed | 1 |
| gripper_open_percentage | 0.4971921443939209 |
| object_held | 0 |
| feet_state_rel_body | [{'x': 0.31900572776794434, 'y': 0.1706952601671219, 'z': -0.5149730443954468}, {'x': 0.31945377588272095, 'y': -0.1728239506483078, 'z': -0.5141311883926392}, {'x': -0.2761070132255554, 'y': 0.16958178579807281, 'z': -0.5163593292236328}, {'x': -0.27343159914016724, 'y': -0.17093735933303833, 'z': -0.5132700800895691}] |
| feet_state_global | [{'x': -0.3417697928292626, 'y': -0.12515192969139824, 'z': -0.5134483088395115}, {'x': -0.29392495738104474, 'y': 0.21502042274777644, 'z': -0.5134433259390588}, {'x': 0.2475817365128402, 'y': -0.20770630519960115, 'z': -0.5168959239815084}, {'x': 0.2928081121510568, 'y': 0.12981321041772212, 'z': -0.5146285409874121}] |
| all_joint_angles | {"fl.hx": 0.00921491626650095, "fl.hy": 0.8005377054214478, "fl.kn": -1.574602723121643, "fr.hx": -0.013359702192246914, "fr.hy": 0.8004810810089111, "fr.kn": -1.5761274099349976, "hl.hx": 0.007037687581032515, "hl.hy": 0.7966209053993225, "hl.kn": -1.5693817138671875, "hr.hx": -0.009716067463159561, "hr.hy": 0.7977815270423889, "hr.kn": -1.581333041191101, "arm0.sh0": 0.0001010894775390625, "arm0.sh1": -3.1184749603271484, "arm0.hr0": 0.0, "arm0.el0": 3.1350982189178467, "arm0.el1": 1.5687037706375122, "arm0.wr0": -0.00045931339263916016, "arm0.wr1": -1.5694420337677002, "arm0.f1x": -0.007805943489074707} |
| all_joint_velocities | {"fl.hx": -0.0014713359996676445, "fl.hy": -0.0019799235742539167, "fl.kn": 0.011371612548828125, "fr.hx": -0.007194998674094677, "fr.hy": 0.0033285804092884064, "fr.kn": -0.01216356735676527, "hl.hx": 0.004889719653874636, "hl.hy": -0.0077947331592440605, "hl.kn": 0.005902839358896017, "hr.hx": 0.01074210461229086, "hr.hy": 0.005369353573769331, "hr.kn": -0.019331036135554314, "arm0.sh0": -0.009795751422643661, "arm0.sh1": 0.011766805313527584, "arm0.hr0": 0.0, "arm0.el0": 0.010913466103374958, "arm0.el1": -0.007954984903335571, "arm0.wr0": 0.004147909115999937, "arm0.wr1": 0.003433068050071597, "arm0.f1x": -0.0011129062622785568} | -->

## RT-1
The RT-1 model from the paper ["RT-1: Robotics Transformer for Real-World Control at Scale"](https://www.roboticsproceedings.org/rss19/p025.pdf) by _Brohan et al._ was modified and fine-tuned on LaNMP.

To be continued...

## ALFRED Seq2Seq
The ALFRED Seq2Seq model from the paper ["ALFRED A Benchmark for Interpreting Grounded Instructions for Everyday Tasks"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shridhar_ALFRED_A_Benchmark_for_Interpreting_Grounded_Instructions_for_Everyday_Tasks_CVPR_2020_paper.pdf) by _Shridhar et al._ was modified and fine-tuned on LaNMP.
This model was trained and ran on an NVIDIA 3090 GPU, so some of the following instructions assume the use of that GPU.

**Preliminary:**
1. Create a Python virtual environment using Python 3.9: `python3.9 -m venv alfred-env`
2. Activate the virtual environment: `source alfred-env/bin/activate`
2. Install and load **CUDA Toolkit 11.8** and **cuDNN 8.7**
3. `cd LaNMP-Dataset/models`
4. `export ALFRED_ROOT=$(pwd)/alfred`
5. `cd alfred`
6. Install all dependencies: `pip install -r requirements.txt`


**Running training:**

The original pretrained model used for fine-tuning can be downloaded from this [Google Drive Folder](https://drive.google.com/drive/folders/12cXF86BgWhWWaMK2EFLbujP2plN4u1ds?usp=sharing). 

1.

```
python models/train/train_seq2seq.py --model seq2seq_im_mask --dout exp/model:{model}_discrete_relative_fold1 --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --pp_data 'data/feats_discrete_relative_fold1' --split_keys 'data/splits/split_keys_discrete_relative_fold1.json --class_mode --relative --preprocess'
```

* `--class_mode` puts the model into classification mode to use cross-entropy loss and output discrete actions
* `--relative` makes the model produce relative (delta between current step and next step) actions rather than global actions
* `--preprocess` preprocesses the data and saves it on disk to be used for the training down the pipeline. This only needs to be ran once. It can be removed after the first time to only run the training.
* More details on all the command-line arguments can be found at `./models/train/train_seq2seq.py`

**Running inference:**

The fine-tuned models can be downloaded from this [Google Drive folder](https://drive.google.com/drive/folders/1Cy1vR64jaYEO5whRO9A2yeQzXyplz1Io?usp=sharing).

The command assumes it is run on a machine with a GUI in order to run the AI2THOR simulator, i.e. not on a headless machine.

1. Place the model pth file in `./models/main_models/alfred/exp`
2. You should already be in the `alfred` directory, and run
```
python models/eval/eval_seq2seq.py --model_path exp/best_test_fold1.pth --gpu --model models.model.seq2seq_im_mask --pp_data data/feats_discrete_relative_fold1 --split_keys 'data/splits/split_keys_discrete_relative_fold1.json'
```
* Any argument that includes 'fold1', that part should be changed to whatever model is being used, e.g. "task" for "best_test_task.pth"
* More details on all the command-line arguments can be found at `./models/eval/eval_seq2seq.py`