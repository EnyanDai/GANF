
# GANF
Offical implementation of "Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series" (ICLR 2022). [[paper]](https://openreview.net/pdf?id=45L_dgP48Vd)

## Requirements

```
torch==1.7.1
```

## Overview

* `./models`: Include the code of **GANF**. (Baselines are also covered for reference)
* `./checkpoint`: The directory to store the model. The trained model for traffic and water system datasets are given in `./checkpoint/eval`
* `./train_water.py` and `./train_traffic.py`: code to train GANF on corresponding datasets
* `./data`: The folder to put the dataset. 


## Datasets
The PMU datasets are proprietary. Thus, in this repo, we only focus on the experiments on the two public datasets: **SWaT** and **Metr-LA**:
* **SWaT**: A water system dataset which can be requested from [iTrust](https://itrust.sutd.edu.sg/). And we utilze the attack_v0 data in Dec/2015 as the whole dataset. You may need firstly transformed the file to .csv to directly use our code. Then, it will be split to train/val/test set in `./dataset.py`. 
* **Metr-LA**: This traffic dataset is only used for exploration experiments which do not require ground-truth outliers. The dataset can be downloader in [here](https://github.com/liyaguang/DCRNN):

## Repreoduce the Results
For training new GANF models on **SWaT**, you can run the bash file:
```
bash train_water.sh
```
A training log on **SWaT** is shown in `./log` as reference to help you reproduce the results.

We also provided trained models in `./checkpoint/eval` for evaluation. You can call:
```
python eval_water.py
```
To train new GANF models on **Metr-LA**, you can simply run:
```
python train_traffic.py
```

## Cite
If you find this repo to be useful, please cite our paper. Thank you.
```
@inproceedings{
dai2022graphaugmented,
title={Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series},
author={Enyan Dai and Jie Chen},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=45L_dgP48Vd}
}
```