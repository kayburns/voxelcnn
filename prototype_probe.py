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
        "--npy_schematic", type=str, default="programmed_houses/house.npy",
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
    model = VoxelCNN(num_features=32)
    predictor = Predictor(model.eval())

    # probe dettings are defined by house shape and origin
    def top():
        np_house = np.load(args.npy_schematic)
        xyz_s = np.vstack(np_house.nonzero()[:-1])
        b_s = np_house[np_house != 0]
        house = np.vstack((b_s, xyz_s))
        origin = (6, 5, 5)
        return house, origin

    def top_no_roof():
        np_house = np.load(args.npy_schematic)
        np_house[6,1:9,1:9,0] = 0 # take off the roof
        xyz_s = np.vstack(np_house.nonzero()[:-1])
        b_s = np_house[np_house != 0]
        house = np.vstack((b_s, xyz_s))
        origin = (6, 5, 5)
        return house, origin

    def side_mid():
        np_house = np.load(args.npy_schematic)
        xyz_s = np.vstack(np_house.nonzero()[:-1])
        b_s = np_house[np_house != 0]
        house = np.vstack((b_s, xyz_s))
        origin = (3, 9, 5)
        return house, origin

    def side_mid_nowall():
        np_house = np.load(args.npy_schematic)
        np_house[:6,9,1:9,0] = 0 # take off the side wall
        xyz_s = np.vstack(np_house.nonzero()[:-1])
        b_s = np_house[np_house != 0]
        house = np.vstack((b_s, xyz_s))
        origin = (3, 9, 5)
        return house, origin

    def side_topright():
        np_house = np.load(args.npy_schematic)
        xyz_s = np.vstack(np_house.nonzero()[:-1])
        b_s = np_house[np_house != 0]
        house = np.vstack((b_s, xyz_s))
        origin = (3, 9, 9)
        return house, origin

    def side_topright_nowall():
        np_house = np.load(args.npy_schematic)
        np_house[:6,9,1:9,0] = 0 # take off the side wall
        xyz_s = np.vstack(np_house.nonzero()[:-1])
        b_s = np_house[np_house != 0]
        house = np.vstack((b_s, xyz_s))
        origin = (3, 9, 9)
        return house, origin

    def botside_nowall():
        np_house = np.load(args.npy_schematic)
        np_house[:6,9,1:9,0] = 0 # take off the side wall
        xyz_s = np.vstack(np_house.nonzero()[:-1])
        b_s = np_house[np_house != 0]
        house = np.vstack((b_s, xyz_s))
        origin = (0, 9, 5)
        return house, origin

    def front_topmid():
        np_house = np.load(args.npy_schematic)
        xyz_s = np.vstack(np_house.nonzero()[:-1])
        b_s = np_house[np_house != 0]
        house = np.vstack((b_s, xyz_s))
        origin = (6, 9, 5)
        return house, origin

    def front_botmid():
        np_house = np.load(args.npy_schematic)
        xyz_s = np.vstack(np_house.nonzero()[:-1])
        b_s = np_house[np_house != 0]
        house = np.vstack((b_s, xyz_s))
        origin = (0, 9, 5)
        return house, origin

    def botmid():
        np_house = np.load(args.npy_schematic)
        np_house[6,1:9,1:9,0] = 0 # take off the roof
        xyz_s = np.vstack(np_house.nonzero()[:-1])
        b_s = np_house[np_house != 0]
        house = np.vstack((b_s, xyz_s))
        origin = (0, 5, 5)
        return house, origin

    prototypes = [(top, 'wall'), (top_no_roof, 'wall'), \
        (side_mid, 'wall'), (side_mid_nowall, 'wall'), \
        (side_topright, 'wall'), (side_topright_nowall, 'wall'), \
        (front_topmid, 'stairs'), (front_botmid, 'fence'), \
        (botmid, 'bookcase'), (botside_nowall, 'floor'), \
        (botmid, 'stairs'), (top_no_roof,'roof')]

    train_conds = ['prims_block_emb_feat32_type5']

    for train_cond in train_conds:

        checkpointer = Checkpointer(os.path.join(args.model_dir, train_cond))
        best_epoch = checkpointer.best_epoch
        checkpointer.load("best", model=model)

        for (schematic, label) in prototypes:
     
            house, origin = schematic()

            # resort based on desired block placement
            def dist_wrapper(origin):
                def dist(xyzb):
                    x, y, z = xyzb[1:]
                    ox, oy, oz = origin
                    return math.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)
                return dist

            house = sorted(house.T, key=dist_wrapper(origin), reverse=True)
            house = np.array(house)
            house_t = torch.tensor(house).long()

            # predict next labels
            checkpointer.load_last_layers(label+'/best', model=model)
            new_blocks = predictor.predict_while_confident(
                house_t, min_steps=10, max_steps=100
            )
            new_house = np.vstack((house, new_blocks)).astype(int)
        
            # reformat to occupancy
            _, max_x, max_y, max_z = np.max(new_house, axis=0)
            _, min_x, min_y, min_z = np.min(new_house, axis=0)
            house_w_prim = np.zeros((max_x-min_x+1, max_y-min_y+1, max_z-min_z+1, 2))
            new_house -= [0, min_x, min_y, min_z]

            # first add old blocks
            n, _ = house.shape
            idxs = tuple(house[:,1:].T.astype(int)) + (np.zeros(n).astype(int),)
            house_w_prim[idxs] = house[:, 0]
            
            # then add new_blocks
            n, _ = new_blocks.shape
            idxs = tuple(new_blocks[:,1:].T) + (np.zeros(n).astype(int),)
            house_w_prim[idxs] = new_blocks[:, 0]

            # save
            save_file_path = os.path.join(
                args.save_dir, "{}_{}_{}_best.npy".format(label, train_cond, schematic.__name__)
            )
            print("Saving to {}".format(save_file_path))
            np.save(save_file_path, house_w_prim)

