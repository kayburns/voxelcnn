import torch
from torch import optim
from torch import nn
from collections import OrderedDict
import os
import sys
import gc
import glob
import pickle as pkl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# added
import datetime
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from voxelcnn.models import VoxelCNN
from voxelcnn.predictor import Predictor
from voxelcnn.checkpoint import Checkpointer
from tqdm import tqdm
import pickle as pkl

##################################### Utils ####################################

def plot_house(pth, house):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*house.nonzero())
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)
    plt.savefig(pth)
    plt.clf()
 
class TransformationBank():

    def __init__(self, pth, load=False):

        self.save_path = os.path.join(pth, 'gen_bank.pkl')
        self.t_dict = {}
        if load:
            self.load_tranformation_bank(self.save_path)

    def load_tranformation_bank(self, pth):
        
        if os.path.exists(pth):
            self.t_dict = pkl.load(open(pth, 'rb'))

    def get_proposal(self, label):

        if label in self.t_dict:
            return self.t_dict[label]
        else:
            return None

    def contains(self, ref_obj):
        return ref_obj in self.t_dict

    def update(self, label, change_vector):
        self.t_dict[label] = change_vector


class GeneratorWrapper():

    def __init__(self):

        save_file_path = '/craftassist/python/VoxelCNN/logs/prims_block_emb_bs_32/'
        self.model = VoxelCNN()
        self.checkpointer = Checkpointer(save_file_path)
        self.best_epoch = self.checkpointer.best_epoch
        self.checkpointer.load("best", model=self.model)
        self.predictor = Predictor(self.model.eval())

    def seen(self, ref_obj):
        """Return true if ref_obj in memory store"""
        return True # TODO

    def update(self, label, blocks, house):

        return # TODO

    def generate_build_proposal(self, ref_blocks, steps=None, no_loc_given=True):
        ref_blocks = torch.tensor([(b[0],) + xyz for (xyz, b) in ref_blocks])
        new_blocks = self.predictor.predict_while_confident(
            ref_blocks, min_steps=10, max_steps=100, no_loc_given=no_loc_given
        )
        new_blocks = [((x, y, z), (b, 0)) for (b, x, y, z) in new_blocks.tolist()]
        if steps:
            new_blocks = new_blocks[:steps]
        return new_blocks
    
    def get_proposal(self, ref_blocks, label, no_loc_given=True):
        self.checkpointer.load_last_layers(label, model=self.model)
        if label=='bed' or label=='door':
            steps = 2
        elif label=='torch':
            steps = 5
        else:
            steps = None

        new_blocks = self.generate_build_proposal(
            ref_blocks, steps=steps, no_loc_given=no_loc_given)
        return new_blocks

