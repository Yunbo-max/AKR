---
dataset_info:
  features:
  - name: 'Unnamed: 0'
    dtype: int64
  - name: path
    dtype: audio
  - name: label
    dtype: string
  splits:
  - name: train
    num_bytes: 908611031.0
    num_examples: 415
  - name: valid
    num_bytes: 310648719.0
    num_examples: 139
  - name: test
    num_bytes: 341896543.0
    num_examples: 139
  - name: train_low
    num_bytes: 211141722.0
    num_examples: 83
  download_size: 1519908309
  dataset_size: 1772298015.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: valid
    path: data/valid-*
  - split: test
    path: data/test-*
  - split: train_low
    path: data/train_low-*
---
# Dataset Card for "beans_dogs"


## Dataset Description

**Paper:** https://doi.org/10.1016/j.anbehav.2003.07.016           

## Dataset Summary

This dataset contains annotated recordings of domestic dog barks with splits and preprocessing like described in BEANS. It is used for **classification** tasks.

## Data Splits

 | train | train_low | valid | test | 
 |------|-----| ------ | ------ | 
 | 415 |  83 | 139 | 139 |