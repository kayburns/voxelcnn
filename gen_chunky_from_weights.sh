train_cond="prims"
echo /craftassist/python/VoxelCNN/prototypes/$train_cond/
echo /craftassist/python/VoxelCNN/logs/$train_cond/
echo /craftassist/test/$train_cond/
which python
python prototype_probe.py \
    --model_dir /craftassist/python/VoxelCNN/logs/$train_cond/ \
    --save_dir /craftassist/python/VoxelCNN/prototypes/$train_cond/
python /craftassist/python/render_schematic.py \
    --npy_schematic /craftassist/python/VoxelCNN/prototypes/$train_cond/ \
    -o /craftassist/test/$train_cond/

