import random

import numpy as np
import torch
from voxelcnn.models import VoxelCNN
from voxelcnn.predictor import Predictor
from voxelcnn.checkpoint import Checkpointer

save_dir = "/scr/kayburns/ckpts/full"

model = VoxelCNN()
model.cuda()
checkpointer = Checkpointer(save_dir)
best_epoch = checkpointer.best_epoch
checkpointer.load("best", model=model)
predictor = Predictor(model.eval())

import pdb; pdb.set_trace()

for i in range(10):
    btype = random.randint(0, 255)
    annotation = torch.tensor([[btype,20,30,0]])
    results = predictor.predict(annotation, steps=10000)

