# Semi-supervised for Depth Estimation from Sparse depth Samples and a Single Image

## Notes
Our network is trained with the KITTI dataset alone, without pretraining on Cityscapes or other similar driving dataset (either synthetic or real). The use of additional data is very likely to further improve the accuracy.

## Requirements
&emsp; &emsp;pytorch 0.4.0  
&emsp; &emsp;cudatoolkit 9.0  
&emsp; &emsp;cudnn 7.1.2  
&emsp; &emsp;torchvision 0.2.1  
&emsp; &emsp;python 3.5.6  

- Download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) Dataset and we use the `2011_09_26_drive_0002_sync`, `2011_09_26_drive_0014_sync`,`2011_09_26_drive_0020_sync`,`2011_09_26_drive_0079_sync`,`2011_09_29_drive_0071_sync`,`2011_09_30_drive_0033_sync`,
`2011_10_03_drive_0042_sync` as val dataset and others as train dataset.

- The code, data and result directory structure is shown as follows
```
├── semi-supervised-depth-completion
├── data
|   ├── kitti_depth
|   |   ├── train
|   |   ├── val
|   └── kitti_rgb
|   |   ├── train
|   |   ├── val
├── results
```

## Training and testing
A complete list of training options is available with 
```bash
python main.py -h
```
For instance,
```bash
python main.py --train-mode dense -b 1 # train with the KITTI semi-dense annotations and batch size 1
python main.py --train-mode sparse+photo # train with the self-supervised framework, not using ground truth
python main.py --resume [checkpoint-path] # resume previous training
python main.py --evaluate [checkpoint-path] # test the trained model
```
