
# Radar3D-Rehab

## A Spatial Millimetre-Wave Radar Point Cloud Dataset for Health Applications

Welcome to the official source code repository of **Radar3D-Rehab**.

This project provides tools for **data processing**, **dataset preparation**, **model training**, and **evaluation** for health-related applications using millimetre-wave radar point clouds.

## Acknowledgements

Many thanks to the following open-source projects. We learned a lot from them and built upon their work:

- [MiliPoint](https://github.com/yizzfz/MiliPoint) - For model building and framework

- [pyKinectAzure](https://github.com/ibaiGorordo/pyKinectAzure/tree/master) - For their excellent Kinect Azure SDK wrapper and examples

We sincerely appreciate the contributions of these projects to the open-source community.
### Clone the repository
```bash
git  clone  https://github.com/gklqmul/Radar3D_Rehab.git
cd  Radar3D_Rehab
```
In this project, there are two main tools including producing data and using data.
``` bash

+---evaluate_data
|---| installpackages.sh
|---| mm
|---| requirements.txt
|---| setup.py
|---+---Reha
|---|---+---dataset
|---|---+---models
|---|---\---session
| |
|---\---configs
|
|---process_data
|---|main.py
|
|---+---additionalfiles  # some processing files for calibration
|---+---class_files
|---+---pykinect_azure  # funciton for Azure Kinect
|---+---utils  # read, save files, draw some plots and so on
|---\---vis    # visualisation
|---| LICENSE
|---| readme.md
\---vis_samples # some graph samples   
```

If only need dataset and models, please jump to [Model Training and Evaluation](#model-training-and-evaluation).

If also want to see more producing details, please jump to [ Data Processing](#data-processing)

## Data Processing

Navigate to the process_data/ directory:

```bash
cd  process_data
```

If you have not recorded any .mkv files yet, please use the command below to collect them. You should also install the required packages as described in this [README](https://github.com/ibaiGorordo/pyKinectAzure/blob/master/README.md).

Now record Kinect video and timestamp files, use:

```bash
python  utils/recordMKV.py
```

This will generate:
A .mkv video file
A timestamps.npy file (used for data alignment)

If you have already recorded .mkv files
Proceed by running:

```bash
python  main.py
```
Through this main, you will get a same dataset folder like ours.

#### Data Processing Pipeline

The five main steps are:

Step 1. Extract skeleton data from .mkv files (grouped by participant).
Step 2. Segment skeleton sequences into actions by identifying the beginning and ending frame IDs.
Step 3. Align radar data with skeleton frame IDs based on timestamps.
Step 4. Split the Kinect and radar data into action-specific segments.
Step 5. Transform skeleton points to radar coordinates.

Note:

1. The raw radar point cloud generation process (signal receiving, initial point cloud creation) is not included. Different radar devices have different mechanisms.
2. Before alignment, ensure you have radar point clouds ready.

#### Dataset Structure

After completing the processing, your dataset will look like:

```bash

/dataset/
├──  env1/  # bright environment
│  └──  subjects/
│  ├──  subject01/  # participant number
│  │  ├──  aligned/  # processed data for models
│  │  │  ├──  action01/
│  │  │  │  ├──  aligned_radar_segment01.h5  # radar point cloud saved in format of frame
│  │  │  │  └──  aligned_skeleton_segment01.npy  # skeleton points from Azure Kinect
│  │  │  ├──  action02/  # second action
│  │  │  └──  ...
│  │  └──  original/  # raw data including unaligned skeleton points and radar point clouds
│  │  ├──  1/  # action groups, include several motions, you can check it through motion energy graphs
│  │  ├──  2/
│  │  └──  ...
│  └──  ...
│  └──  subject26
├──  env2/  # low-light environment
│  └──  subjects/
│  └──  subject01/
│  └──  subject02/
│  └──  ...

```

Each subject contains 21 kinds of actions.


## Model Training and Evaluation

### Download the dataset

Download the processed dataset (approximately 900MB) from:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YKS954

Unzip it in the same path as the cloned repository.
Navigate to the evaluate_data/ directory:

```bash
cd  evaluate_data
```

Install required packages:

```bash
pip  install  .
```

### Commands

Command Format

```bash
./mm <train/test> <task> <model> <args>
```

Note: On Windows, use "python mm" instead of "./mm".


Example: Training

```bash
./mm  train  radar_kp  dgcnn  \
--save session_name \
-config  ./configs/keypoint.toml  \
-a gpu
```

Override configuration values from the command line:

```bash
./mm  train  radar_kp  mlp  \
--save session_name \
-config  ./configs/keypoint.toml  \
--dataset_stacks 5
```

Example: Testing

```bash
./mm  test  radar_kp  dgcnn  \
--load session_name \
-config  ./configs/keypoint.toml
```

### Hyperparameters

--dataset_stacks / stacks: Number of frames stacked together with the current frame.

--dataset_zero_padding / zero_padding: Padding strategy:

data_point — pad each frame individually.

stack — pad the entire stack at once.

Other hyperparameters are available in the .toml configuration files inside configs/.

### Models

Available models:

DGCNN
PointNet++
PointTransformer

Their performances are reported in the accompanying paper.

Tasks

radar_kp - Keypoint Estimation
Input (X): Radar point cloud (n, 3)
Output (Y): 32 skeleton keypoints

Train

```bash
python  mm  train  radar_kp  dgcnn  \
--save radar_dgcnn \
-config  ./configs/keypoint.toml  \
-w 0 \
-a  gpu  \
-m 300 \
-n  -1
```

Test
```bash
python  mm  test  radar_kp  dgcnn  \
--load radar_dgcnn \
-config  ./configs/keypoint.toml  \
-w 0 \
-a  gpu
```

radar_iden - Subject Identification
Input (X): Radar point cloud (n, 3)

Output (Y): Subject ID (1–26)

Train
```bash

python  mm  train  radar_iden  dgcnn  \
--save radar_iden_dgcnn \
-config  ./configs/identity.toml  \
-w 0 \
-a  gpu  \
-m 300 \
-n  -1
```

Test
```bash
python  mm  test  radar_iden  dgcnn  \
--load radar_iden_dgcnn \
-config  ./configs/identity.toml  \
-w 0 \
-a  gpu
```
radar_act - Action Classification
Input (X): Radar point cloud (n, 3)
Output (Y): Action ID (1–21)

Train
```bash

python  mm  train  radar_act  dgcnn  \
--save radar_act_dgcnn \
-config  ./configs/action.toml  \
-w 0 \
-a  gpu  \
-m 300 \
-n  -1
```

Test
```bash
python  mm  test  radar_act  dgcnn  \
--load radar_act_dgcnn \
-config  ./configs/action.toml  \
-w 0 \
-a  gpu
```
### Citation
If you find this project useful, please consider citing: