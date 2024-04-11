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
    parser.add_argument('--data', help='data folder', default='/users/ajaafar/data/shared/lanmp/lanmp_dataset.hdf5') #make relative later
    parser.add_argument('--pp_data', help='preprocessed dataset folder', default='/users/ajaafar/data/ajaafar/NPM-Dataset/models/main_models/alfred/data/feats') #make relative later
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    # parser.add_argument('--skip_existing', help='skip folders that already have feats', action='store_true')
    parser.add_argument('--visual_model', default='resnet18', help='model type: maskrcnn or resnet18', choices=['maskrcnn', 'resnet18'])
    parser.add_argument('--filename', help='filename of feat', default='feat_conv.pt')

    # parser
    args = parser.parse_args()

    # load resnet model
    extractor = Resnet(args, eval=True)

    # Open the HDF5 file
    with h5py.File(args.data, 'r') as hdf:
        # Iterate through all top-level groups (arbitrary groups) in the HDF5 file
        for top_group_name in tqdm(hdf.keys(), desc='Trajectory'):
            top_group = hdf[top_group_name]
            
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
                            raw_imgs.append(rgb_data)
                        

            imgs = [Image.fromarray(raw_img) for raw_img in raw_imgs]
            feat = extractor.featurize(imgs, batch=len(imgs))
            path = os.path.join(args.pp_data, top_group_name, 'pp', args.filename)
            torch.save(feat.cpu(), path)