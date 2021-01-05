import argparse
import numpy as np

from .predictor import Predictor
from .checkpoint import Checkpointer
from .models import VoxelCNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="examine and inspect prototypes from each label generator"
    )
    parser.add_argument("--npy_schematic", type=str, help="Path to npy house")
    args = parser.parse_args()

    # model setup
    save_file_path = '/craftassist/python/VoxelCNN/logs/'
    model = VoxelCNN()
    checkpointer = Checkpointer(save_file_path)
    best_epoch = checkpointer.best_epoch
    checkpointer.load("best", model=model)a
    predictor = Predictor(model.eval())

    # load schematic
    np_house = np.load(args.npy_schematic)
    xyz_s = np.vstack(np_house.nonzero()[:-1])
    b_s = np_house[np_house != 0]
    house = np.vstack(b_s, xyz_s)

    for label in labels:
        checkpointer.load_last_layers(label, model=model)
        new_blocks = predictor.predict_while_confident(
            house, min_steps=10, max_steps=100
        )

