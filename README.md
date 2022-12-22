
# GANF
Offical implementation of "Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series" (ICLR 2022). [[paper]](https://openreview.net/pdf?id=45L_dgP48Vd)

## Requirements

```
torch==1.7.1
```

## Overview

* `./models`: This directory includes the code of GANF as well as basline methods.
* `./checkpoint`: This directory stores the trained models. The trained models for the datasets **SWaT** and **Metr-LA** are given in `./checkpoint/eval`.
* `./train_water.py` and `./train_traffic.py`: These programs are used to train GANF on the corresponding datasets.
* `./data`: This directory is used to store the datasets.


## Datasets
The paper uses three datasets for experiments:
* **SWaT**: This water system dataset can be requested from [iTrust](https://itrust.sutd.edu.sg/). We utilze the attack_v0 data in Dec/2015 for experimentation. You may need to first convert the file format to .csv to use our code. Then, use `./dataset.py` to perform train/val/test split.
* **Metr-LA**: This traffic dataset does not include ground-truth outliers. It can be used for exploratory studies of density estimation. The dataset can be downloaded from [this GitHub](https://github.com/liyaguang/DCRNN).
* **PMU**: This power grid dataset is proprietary and we are unable to offer it for public use.

## Experiments
To train a GANF model on **SWaT**, run the bash script:
```
bash train_water.sh
```
The training log will be located at `./log` as a reference to reproduce the results in the paper.

We also provide trained models in `./checkpoint/eval` for evaluation. You can call:
```
python eval_water.py
```

To train a GANF model on **Metr-LA**, run:
```
python train_traffic.py
```

## Citation
If you find this repo useful, please cite the paper. Thank you!
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
