import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import torch
import os
import h5py
from PIL import Image
from tqdm import tqdm
from nn.resnet import Resnet
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--data', help='data folder', default='../../../dataset/sim_dataset.hdf5') 
    parser.add_argument('--pp_data', help='preprocessed dataset folder', default='data/vis_feats')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--visual_model', default='resnet18', help='model type: maskrcnn or resnet18', choices=['maskrcnn', 'resnet18'])
    parser.add_argument('--filename', help='filename of feat', default='feat_conv.pt')

    # parser
    args = parser.parse_args()

    if not os.path.exists(args.pp_data):
        os.makedirs(args.pp_data)

    # load resnet model
    extractor = Resnet(args, eval=True)

    # Open the HDF5 file
    with h5py.File(args.data, 'r') as hdf:
        # Iterate through all top-level groups (arbitrary groups) in the HDF5 file
        for top_group_name in tqdm(hdf.keys(), desc='Trajectory'):
            top_group = hdf[top_group_name]
            folder_path = os.path.join(args.pp_data, top_group_name, 'pp')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_path = os.path.join(args.pp_data, top_group_name, 'pp', args.filename)
            
            #skipping already done ones
            if os.path.isfile(file_path):
                print(f"Skipping {top_group_name}")
                continue
            
            raw_imgs = []
            # Iterate through all sub-groups within the top-level group (folder_[NUM] groups)
            for folder_name in tqdm(top_group.keys(), desc='Time-step'):
                # Check if the sub-group name matches the 'folder_[NUM]' pattern
                if folder_name.startswith('folder_'):
                    folder_group = top_group[folder_name]
                    
                    # datasets that match the rgb_[NUM] pattern
                    for dataset_name in folder_group.keys():
                        if dataset_name.startswith('rgb_'):
                            # Access the rgb dataset
                            rgb_dataset = folder_group[dataset_name]
                            rgb_data = rgb_dataset[:]
                            #verification of images
                            #image = Image.fromarray(rgb_data, 'RGB')
                            #image.show()
                            raw_imgs.append(rgb_data)
                        

            imgs = [Image.fromarray(raw_img) for raw_img in raw_imgs]
            feat = extractor.featurize(imgs, batch=len(imgs))
            torch.save(feat.cpu(), file_path)
