# ST-TCN
Code for DASFAA 2024 submission 1026: "Flexible Contact Correlation Learning on Spatio-Temporal Trajectories"
### Requirements
- torch
- torch_geometric
- networkx
- geopy
- numba
- transbigdata
### Preprocessing
- Step1: Download data from https://www.cs.rutgers.edu/~dz220/data.html.
- Step2: Put the data file in <tt>../datasets/sz/</tt>, and unzip it as <tt>TaxiData.txt</tt>.
- Step3: Run
```
mkdir -p data/sz
python preprocess_sz.py
```
### Generating ground truth
```
mkdir data/sz/s_t temp_data
python ground_truth.py --dataset sz --contact_factor 3 --score_type s_t
```
Here, `contact_factor` and `score_type` are introduced in the Preliminaries section.
### Training
```
mkdir logs models_pth
python pretrain.py --dataset sz
python main.py --dataset sz --contact_factor 3 --score_type s_t --lr 3e-4 --mode train
```
### Testing
```
python main.py --dataset sz --contact_factor 3 --score_type s_t --mode test
```
