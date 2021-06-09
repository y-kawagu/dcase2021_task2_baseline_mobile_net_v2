# dcase2021_task2_mobile_net_v2
MobileNetV2-based baseline system for [DCASE2021 Challenge Task 2](http://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds).

## Description
This system consists of two main scripts:
- `00_train.py`
  - "Development" mode: 
    - This script trains a model for each machine type by using the directory `dev_data/<machine_type>/train/`.
  - "Evaluation" mode: 
    - This script trains a model for each machine type by using the directory `eval_data/<machine_type>/train/`. (This directory will be from the "additional training dataset".)
- `01_test.py`
  - "Development" mode:
    - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `dev_data/<machine_type>/source_test/` and `dev_data/<machine_type>/target_test/`.
    - The csv files are stored in the directory `result/`.
    - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
  - "Evaluation" mode: 
    - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `eval_data/<machine_type>/source_test/` and `eval_data/<machine_type>/target_test/`. (These directories will be from the "evaluation dataset".)
    - The csv files are stored in the directory `result/`.

## Usage

### 1. Clone repository
Clone this repository from Github.

### 2. Download datasets
We will launch the datasets in three stages. 
So, please download the datasets in each stage:
- "Development dataset"
  - Download `dev_data_<machine_type>.zip` from https://zenodo.org/record/4562016. 
- "Additional training dataset", i.e. the evaluation dataset for training
  - After April. 1, 2021, download `eval_data_train_<machine_type>.zip` from https://zenodo.org/record/4660992.
- "Evaluation dataset", i.e. the evaluation dataset for test
  - After June. 1, 2021, download `eval_data_test_<machine_type>.zip` from https://zenodo.org/record/4884786.

### 3. Unzip dataset
Unzip the downloaded files and make the directory structure as follows:
- /dcase2021_task2_mobile_net_v2
    - /00_train.py
    - /01_test.py
    - /common.py
    - /keras_model.py
    - /baseline.yaml
    - /readme.md
- /dev_data
    - /fan
        - /train (Normal data in the **source** and **target** domains for all sections are included.)
            - /section_00_source_train_normal_0000_<attribute>.wav
            - ...
            - /section_00_source_train_normal_0999_<attribute>.wav
            - /section_00_target_train_normal_0000_<attribute>.wav
            - /section_00_target_train_normal_0001_<attribute>.wav
            - /section_00_target_train_normal_0002_<attribute>.wav
            - /section_01_source_train_normal_0000_<attribute>.wav
            - ...
            - /section_02_target_train_normal_0002_<attribute>.wav
        - /source_test (Normal and anomaly data in the **source** domain for all sections are included.)
            - /section_00_source_test_normal_0000.wav
            - ...
            - /section_00_source_test_normal_0099.wav
            - /section_00_source_test_anomaly_0000.wav
            - ...
            - /section_00_source_test_anomaly_0099.wav
            - /section_01_source_test_normal_0000.wav
            - ...
            - /section_02_source_test_anomaly_0099.wav
        - /target_test (Normal and anomaly data in the **target** domain for all sections are included.)
            - /section_00_target_test_normal_0000.wav
            - ...
            - /section_00_target_test_normal_0099.wav
            - /section_00_target_test_anomaly_0000.wav
            - ...
            - /section_00_target_test_anomaly_0099.wav
            - /section_01_target_test_normal_0000.wav
            - ...
            - /section_02_target_test_anomaly_0099.wav
    - /gearbox (The other machine types have the same directory structure as fan.)
    - /pump
    - /slider
    - /valve
    - /ToyCar
    - /ToyTrain
- /eval_data (Add this directory after launch)
    - /fan
        - /train (Unzipped additional training dataset. Normal data in the **source** and **target** domains for all sections are included.)
            - /section_03_source_train_normal_0000_<attribute>.wav
            - ...
            - /section_03_source_train_normal_0999_<attribute>.wav
            - /section_03_target_train_normal_0000_<attribute>.wav
            - /section_03_target_train_normal_0001_<attribute>.wav
            - /section_03_target_train_normal_0002_<attribute>.wav
            - /section_04_source_train_normal_0000_<attribute>.wav
            - ...
            - /section_05_target_train_normal_0002_<attribute>.wav
        - /source_test (Unzipped evaluation dataset. Normal and anomaly data in the **source** domain for all sections are included.)
            - /section_03_source_test_0000.wav
            - ...
            - /section_03_source_test_0199.wav
            - /section_04_source_test_0000.wav
            - ...
            - /section_05_source_test_0199.wav
        - /target_test (Unzipped evaluation dataset. Normal and anomaly data in the **target** domain for all sections are included.)
            - /section_03_target_test_0000.wav
            - ...
            - /section_03_target_test_0199.wav
            - /section_04_target_test_0000.wav
            - ...
            - /section_05_target_test_0199.wav
    - /gearbox (The other machine types have the same directory structure as fan.)
    - /pump
    - /slider
    - /valve
    - /ToyCar
    - /ToyTrain

### 4. Change parameters
You can change parameters for feature extraction and model definition by editing `baseline.yaml`.

### 5. Run training script (for the development dataset)
Run the training script `00_train.py`. 
Use the option `-d` for the development dataset `dev_data/<machine_type>/train/`.
```
$ python3.6 00_train.py -d
```
Options:

| Argument                    |                                   | Description                                                  | 
| --------------------------- | --------------------------------- | ------------------------------------------------------------ | 
| `-h`                        | `--help`                          | Application help.                                            | 
| `-v`                        | `--version`                       | Show application version.                                    | 
| `-d`                        | `--dev`                           | Mode for the development dataset                             |  
| `-e`                        | `--eval`                          | Mode for the additional training and evaluation datasets     | 

`00_train.py` trains a model for each machine type and store the trained models in the directory `model/`.

### 6. Run test script (for the development dataset)
Run the test script `01_test.py`.
Use the option `-d` for the development dataset `dev_data/<machine_type>/test/`.
```
$ python3.6 01_test.py -d
```
The options for `01_test.py` are the same as those for `00_train.py`.
`01_test.py` calculates an anomaly score for each wav file in the directories `dev_data/<machine_type>/source_test/` and `dev_data/<machine_type>/target_test/`.
A csv file for each section including the anomaly scores will be stored in the directory `result/`.
If the mode is "development", the script also outputs another csv file including AUC, pAUC, precision, recall, and F1-score for each section.

### 7. Check results
You can check the anomaly scores in the csv files `anomaly_score_<machine_type>_section_<section_index>_<domain>_test.csv` in the directory `result/`.
Each anomaly score corresponds to a wav file in the directories `dev_data/<machine_type>/source_test/` and `dev_data/<machine_type>/target_test/`:

`anomaly_score_fan_section_00_source_test.csv`
```
section_00_source_test_normal_0000.wav	-5.492875
section_00_source_test_normal_0001.wav	-13.004328
section_00_source_test_normal_0002.wav	-7.7093716
section_00_source_test_normal_0003.wav	-6.13771
section_00_source_test_normal_0004.wav	-4.9352393
section_00_source_test_normal_0005.wav	-3.93735
  ...
```

Also, anomaly detection results after thresholding can be checked in the csv files `anomaly_score_<machine_type>_section_<section_index>_<domain>_test.csv`:

`decision_result_fan_section_00_source_test.csv`
```
section_00_source_test_normal_0000.wav	0
section_00_source_test_normal_0001.wav	0
section_00_source_test_normal_0002.wav	0
section_00_source_test_normal_0003.wav	0
section_00_source_test_normal_0004.wav	0
section_00_source_test_normal_0005.wav	1
  ...
```

Also, you can check performance indicators such as AUC, pAUC, precision, recall, and F1 score:

`result.csv`
```  
fan						
section	domain	AUC	pAUC	precision	recall	F1 score
0	source	0.4749	0.525263158	0.75	0.03	0.057692308
1	source	0.8041	0.796842105	1	0.18	0.305084746
2	source	0.7661	0.771578947	0.964285714	0.54	0.692307692
0	target	0.5651	0.576315789	1	0.1	0.181818182
1	target	0.75	0.62	0.840909091	0.37	0.513888889
2	target	0.6067	0.6	1	0.08	0.148148148
arithmetic mean		0.66115	0.648333333	0.925865801	0.216666667	0.316489994
harmonic mean		0.63790168	0.633610854	0.914695559	0.090987059	0.165510386
  ...
valve						
section	domain	AUC	pAUC	precision	recall	F1 score
0	source	0.615	0.556842105	1	0.01	0.01980198
1	source	0.5735	0.501052632	1	0.01	0.01980198
2	source	0.5833	0.516842105	0	0	0
0	target	0.5391	0.515789474	0	0	0
1	target	0.7371	0.617894737	0.607407407	0.82	0.69787234
2	target	0.5277	0.507894737	0	0	0
arithmetic mean		0.59595	0.536052632	0.434567901	0.14	0.122912717
harmonic mean		0.588771731	0.533212354	4.44E-16	4.44E-16	4.44E-16
						
		AUC	pAUC	precision	recall	F1 score
arithmetic mean over all machine types, sections, and domains		0.616885065	0.571176451	0.485820851	0.203142415	0.220250915
harmonic mean over all machine types, sections, and domains		0.59239137	0.559521566	6.22E-16	6.22E-16	6.22E-16
```

### 8. Run training script for the additional training dataset (after April 1, 2021)
After the additional training dataset is launched, download and unzip it.
Move it to `eval_data/<machine_type>/train/`.
Run the training script `00_train.py` with the option `-e`. 
```
$ python3.6 00_train.py -e
```
Models are trained by using the additional training dataset `eval_data/<machine_type>/train/`.

### 9. Run test script for the evaluation dataset (after June 1, 2021)
After the evaluation dataset for test is launched, download and unzip it.
Move it to `eval_data/<machine_type>/source_test/` and `eval_data/<machine_type>/target_test/`.
Run the test script `01_test.py` with the option `-e`. 
```
$ python3.6 01_test.py -e
```
Anomaly scores are calculated using the evaluation dataset, i.e., `eval_data/<machine_type>/source_test/` and `eval_data/<machine_type>/target_test/`.
The anomaly scores are stored as csv files in the directory `result/`.
You can submit the csv files for the challenge.
From the submitted csv files, we will calculate AUC, pAUC, and your ranking.

## Dependency
We develop the source code on Ubuntu 16.04 LTS and 18.04 LTS.
In addition, we checked performing on **Ubuntu 16.04 LTS**, **18.04 LTS**, **CentOS 7**, and **Windows 10**.

### Software packages
- p7zip-full
- Python == 3.6.5
- FFmpeg

### Python packages
- Keras                         == 2.3.0
- Keras-Applications            == 1.0.8
- Keras-Preprocessing           == 1.0.5
- matplotlib                    == 3.0.3
- numpy                         == 1.18.1
- PyYAML                        == 5.1
- scikit-learn                  == 0.22.2.post1
- scipy                         == 1.1.0
- librosa                       == 0.6.0
- audioread                     == 2.1.5 (more)
- setuptools                    == 41.0.0
- tensorflow                    == 1.15.0
- tqdm                          == 4.43.0

## Citation
If you use this baseline system, please cite all the following three papers:
- Yohei Kawaguchi, Keisuke Imoto, Yuma Koizumi, Noboru Harada, Daisuke Niizumi, Kota Dohi, Ryo Tanabe, Harsh Purohit, and Takashi Endo, "Description and Discussion on DCASE 2021 Challenge Task 2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring under Domain Shifted Conditions," in arXiv e-prints: 2106.04492, 2021. [URL](https://arxiv.org/abs/2106.04492)
- Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, Shoichiro Saito, "ToyADMOS2: Another Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection under Domain Shift Conditions," in arXiv e-prints: 2106.02369, 2021. [URL](https://arxiv.org/abs/2106.02369)
- Ryo Tanabe, Harsh Purohit, Kota Dohi, Takashi Endo, Yuki Nikaido, Toshiki Nakamura, and Yohei Kawaguchi, "MIMII DUE: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection with Domain Shifts due to Changes in Operational and Environmental Conditions," in arXiv e-prints: 2105.02702, 2021. [URL](https://arxiv.org/abs/2105.02702)
