# Robustness Evaluation of Deep Unsupervised Learning Algorithms for Intrusion Detection Systems
This repository collects different unsupervised machine learning algorithms to detect anomalies.
## Implemented models
We have implemented the following models. Our implementations of ALAD, DeepSVDD, 
DROCC and MemAE closely follows the original implementations already available on GitHub.
- [x] [AutoEncoder]()
- [x] [ALAD](https://arxiv.org/abs/1812.02288)
- [x] [DAGMM](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf)
- [x] [DSEBM](https://arxiv.org/abs/1605.07717)
- [x] [DUAD](https://openaccess.thecvf.com/content/WACV2021/papers/Li_Deep_Unsupervised_Anomaly_Detection_WACV_2021_paper.pdf)
- [x] [NeuTraLAD](https://arxiv.org/pdf/2103.16440.pdf)

## Dependencies
A complete dependency list is available in requirements.txt.
We list here the most important ones:
- torch@1.10.2 with CUDA 11.3
- numpy
- pandas
- scikit-learn

## Installation
Assumes latest version of Anaconda was installed.
```
$ conda create --name [ENV_NAME] python=3.8
$ conda activate [ENV_NAME]
$ pip install -r requirements.txt
```
Replace `[ENV_NAME]` with the name of your environment.

## Usage
From the root of the project.
```
$ python -m src.main 
-m [model_name]
-d [/path/to/dataset/file.{npz,mat}]
--dataset [dataset_name]
--batch-size [batch_size]
```

Our model contains the following parameters:
- `-m`: selected machine learning model (**required**)
- `-d`: path to the dataset (**required**)
- `--batch-size`: size of a training batch (**required**)
- `--dataset`: name of the selected dataset. Choices are `Arrhythmia`, `KDD10`, `IDS2018`, `NSLKDD`, `USBIDS`, `Thyroid` (**required**).
- `-e`: number of training epochs (default=200)
- `--n-runs`: number of time the experiment is repeated (default=1)
- `--lr`: learning rate used during optimization (default=1e-4)
- `--pct`: percentage of the original data to keep (useful for large datasets, default=1.)
- `rho`: anomaly ratio within the training set (default=0.)
- `--results-path`: path where the results are stored (default="../results")
- `--model-path`: path where models will be stored (default="../models")
- `--test-mode`: loads models from `--model_path` and tests them (default=False)
- `--hold_out`: Percentage of anomalous data to holdout for possible contamination of the training set (default=0)
- `--rho`: Contamination ratio of the training set(default=0)

Please note that datasets must be stored in `.npz` or `.mat` files. Use the preprocessing scripts within `data_process`
to generate these files.

## Example
To train a DAGMM on the KDD 10 percent dataset with the default parameters described in the original paper:
```
$ python  -m src.main -m DAGMM -d [/path/to/dataset.npz] --dataset KDD10 --batch-size 1024 --results-path ./results/KDD10 --models-path ./models/KDD10
```
Replace `[/path/to/dataset.npz]` with the path to the dataset in a numpy-friendly format.

