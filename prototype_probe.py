import math
import os
import argparse
from glob import glob
import numpy as np
import torch

from voxelcnn.predictor import Predictor
from voxelcnn.checkpoint import Checkpointer
from voxelcnn.models import VoxelCNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="examine and inspect prototypes from each label generator"
    )
    parser.add_argument(
        "--npy_schematics", type=str, default="programmed_houses/",
        help="Path to npy house"
    )
    parser.add_argument(
        "--model_dir", type=str, default="/craftassist/python/VoxelCNN/logs/",
        help="directory with models per label"
    )
 
    parser.add_argument("--debug", action='store_true')
    parser.add_argument(
        "--save_dir", type=str, default="./prototypes",
        help="Path to save visualizations"
    )
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    # model setup
    model = VoxelCNN()
    checkpointer = Checkpointer(args.model_dir)
    best_epoch = checkpointer.best_epoch
    checkpointer.load("best", model=model)
    predictor = Predictor(model.eval())

    #for path in os.listdir(args.npy_schematics):
    fpath = '/craftassist/minecraft_specs/schematics/cleaned_houses/'
    for val_home in [98, 30, 33, 34, 16, 44, 138, 14, 154, 158, 159, 165, 17, 173, 194, 197, 199, 113, 114, 116, 125, 126]:
        npy_file = os.path.join(fpath, "validation{}.npy".format(val_home))

        # load schematic
        np_house = np.load(npy_file)
        #np_house[6,:9,:9,0] = 0 # take off the roof
        #np_house[:7,:10,9,0] = 0 # take off a wall
        #np_house[4,9,9,0] = 5 # add suggestion

        # convert npy to block list format
        xyz_s = np.vstack(np_house.nonzero()[:-1])
        b_s = np_house[np_house != 0]
        house = np.vstack((b_s, xyz_s))

        # re-sort based on desired block placement
        def dist(xyzb):
            x, y, z = xyzb[1:]
            ox, oy, oz = [100,100,100]
            return math.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)

        house = sorted(house.T, key=dist, reverse=True)
        house = np.array(house)
        house_t = torch.tensor(house).long()

        # fetch label names from log dir
        label_format = os.path.join(args.model_dir, '*/')
        labels = glob(label_format)
        labels = [lbl.split('/')[-2] for lbl in labels]

        labels=["wall", "roof"]
        for label in labels:

            # predict next labels
            checkpointer.load_last_layers(label, model=model)
            new_blocks = predictor.predict_while_confident(
                house_t, min_steps=10, max_steps=40
            )

            import pdb; pdb.set_trace()
            # zero out duplicates, bc dense() sums over multiple entries
            new_blocks = torch.tensor(new_blocks).long()
            for block in new_blocks:
                block_c = block[1:]
                if np.all(block_c == new_blocks)
            idxs = house_t[:,1:].unsqueeze(1).repeat((1, 40, 1))
            idxs = idxs[:,:,:] == new_blocks[:,1:]
            idxs = torch.any(torch.all(idxs, dim=2), dim=1)
            house_t[idxs,0] = 0
            new_house = torch.cat((house_t, new_blocks), dim=0)
            house_t = new_house
            
        new_house = new_house.numpy()

        # reformat to occupancy
        _, max_x, max_y, max_z = np.max(new_house, axis=0)
        _, min_x, min_y, min_z = np.min(new_house, axis=0)
        house_w_prim = np.zeros((max_x-min_x+1, max_y-min_y+1, max_z-min_z+1, 2))
        new_house -= [0, min_x, min_y, min_z]

        # first add old blocks
        idxs = tuple(house[:,1:].T.astype(int)) + (np.zeros(n).astype(int),)
        house_w_prim[idxs] = house[:, 0]
        
        # then add new_blocks
        n, _ = new_blocks.shape
        idxs = tuple(new_blocks[:,1:].T) + (np.zeros(n).astype(int),)
        import pdb; pdb.set_trace()
        house_w_prim[idxs] = new_blocks[:, 0]

        # save
        save_file_path = os.path.join(args.save_dir, "{}.npy".format(val_home))
        np.save(save_file_path, house_w_prim)

