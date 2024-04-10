import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import torch
import os
from PIL import Image
from nn.resnet import Resnet
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--data', help='data folder', default='/users/ajaafar/data/shared/lanmp/lanmp_dataset.hdf5')
    parser.add_argument('--batch', help='batch size', default=256, type=int)
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--skip_existing', help='skip folders that already have feats', action='store_true')
    parser.add_argument('--visual_model', default='resnet18', help='model type: maskrcnn or resnet18', choices=['maskrcnn', 'resnet18'])
    parser.add_argument('--filename', help='filename of feat', default='feat_conv.pt')
    parser.add_argument('--img_folder', help='folder containing raw images', default='raw_images')

    # parser
    args = parser.parse_args()

    # load resnet model
    extractor = Resnet(args, eval=True)
    skipped = []

    import h5py

    # Path to your HDF5 file
    # hdf5_path = '/users/ajaafar/data/shared/lanmp/lanmp_dataset.hdf5'

    # Open the HDF5 file
    with h5py.File(args.data, 'r') as hdf:
        # Iterate through all groups in the HDF5 file
        for group_name in hdf.keys():
            # Check if the group name matches the 'folder_[NUM]' pattern
            if group_name.startswith('folder_'):
                # Construct the name of the rgb dataset based on the folder name
                rgb_name = f"rgb_{group_name.split('_')[1]}"
                
                # Check if the rgb dataset exists in the current group
                if rgb_name in hdf[group_name]:
                    # Access the rgb dataset
                    rgb_dataset = hdf[group_name][rgb_name]
                    breakpoint()
                    
                    # Set it to a variable (assuming you want to work with the data)
                    rgb_data = rgb_dataset[:]
                    
                    # Now you can work with `rgb_data`
                    # For example, print the shape of the dataset
                    print(f"Shape of {rgb_name} in {group_name}: {rgb_data.shape}")
                    # Note: This example prints the shape of the rgb dataset for demonstration. 
                    # You can modify this part to perform other operations with `rgb_data`.







    for root, dirs, files in os.walk(args.data):
        if os.path.basename(root) == args.img_folder:
            fimages = sorted([os.path.join(root, f) for f in files
                              if (f.endswith('.png') or (f.endswith('.jpg')))])
            if len(fimages) > 0:
                if args.skip_existing and os.path.isfile(os.path.join(root, 'feat_conv.pt')):
                    continue
                try:
                    print('{}'.format(root))
                    image_loader = Image.open if isinstance(fimages[0], str) else Image.fromarray
                    images = [image_loader(f) for f in fimages]
                    feat = extractor.featurize(images, batch=args.batch)
                    torch.save(feat.cpu(), os.path.join(root.replace('raw_images', ''), args.filename))
                except Exception as e:
                    print(e)
                    print("Skipping " + root)
                    skipped.append(root)

            else:
                print('empty; skipping {}'.format(root))

    print("Skipped:")
    print(skipped)