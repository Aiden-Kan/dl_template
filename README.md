# Dl_template

This is the template of deep-learning for better use, which is referred to DL-boilerplate-master from Lu Yuchen.

## Folder Structure
```
├── checkpoint/
│   └── epochs/                 # save checkpoints for every epoch
├── data/                       # raw data 
│   ├── train
│   ├── val            
│   └── test           
├── dataset/
│   └── dataset.py              # one task corresponds to one dataset
├── models/
│   └── xxNet.py                # one model corresponds to one file
├── logs/                       # record different data with tensorboard
├── runs/                       # output
├── utils/
│   ├── measure.py              # different indexes to access the performance
│   └── ...
├── test.py
└── train.py
```
## To be updating for better use ...