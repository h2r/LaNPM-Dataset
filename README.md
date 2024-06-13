# NPM-Dataset
A comprehensive robotics dataset that includes navigation, perception, and manipulation data per data point.

## Dataset format
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

Under each folder, there are three main files: `depth_<num>`, `inst_seg_<num>`, and `rgb_<num>`,
which correspond to the depth image, segmentation image, and rgb image, respectively.

Under the metadata for each folder, there is a dumped json describing other metadata of each time step.
The metadata include the following:

| key | description | value |
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
| hand_sphere_radius | Radius of simulated hand sphere | 0.05999999865889549 |


### Real Dataset
