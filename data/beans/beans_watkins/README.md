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
    num_bytes: 791557202.69
    num_examples: 1017
  - name: valid
    num_bytes: 272364369.0
    num_examples: 339
  - name: test
    num_bytes: 269221587.0
    num_examples: 339
  - name: train_low
    num_bytes: 222944097.0
    num_examples: 203
  download_size: 1428752044
  dataset_size: 1556087255.69
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
# Dataset Card for "beans_watkins"

## Dataset Description

**Paper:** 
https://doi.org/10.1121/2.0000358
          
## Dataset Summary

This dataset contains annotated recordings of marine mammal sounds with splits and preprocessing like described in BEANS. It is used for **classification** tasks.

## Data Splits

 | train | train_low | valid | test | 
 |------|-----| ------ | ------ | 
 | 1017 |  203 | 339 | 339 |