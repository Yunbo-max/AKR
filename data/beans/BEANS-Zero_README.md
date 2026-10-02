---
license: other
task_categories:
  - audio-classification
language:
  - en
tags:
  - biology
  - bioacoustics
  - audio-classification
  - multimodal
  - zero-shot
pretty_name: BEANS-Zero
size_categories:
- 100K<n<1M
configs:
  - config_name: "BEANS-Zero"
    data_files:
      - split: test
        path: shard*
---

# BEANS-Zero

**Version:** 0.1.0
**Created on:** 2025-04-12
**Creators:**
- Earth Species Project (https://www.earthspecies.org)

## Overview

BEANS-Zero is a **bioacoustics** benchmark designed to evaluate multimodal audio-language models in zero-shot settings. Introduced in the paper NatureLM-audio paper ([Robinson et al., 2025](https://openreview.net/forum?id=hJVdwBpWjt)), it brings together tasks from both existing datasets and newly curated resources.

The benchmark focuses on models that take a bioacoustic audio input (e.g., bird or mammal vocalizations) and a text instruction (e.g., "What species is in this audio?"), and return a textual output (e.g., "Taeniopygia guttata"). As a zero-shot benchmark, BEANS-Zero contains only a test split—no training or in-context examples are provided.

Many tasks originate from the original [BEANS benchmark](https://arxiv.org/abs/2210.12300), but BEANS-Zero adds new datasets and task types that broaden the evaluation scope.

## Tasks and Applications

BEANS-Zero supports a wide range of zero-shot evaluation tasks, including:
- **Audio Classification** — Identify species or sound categories from animal vocalizations.
- **Audio Detection** — Detect the presence of species in long-form recordings.
- **Audio Captioning** — Generate natural language descriptions of acoustic scenes.

## Dataset Composition

BEANS-Zero combines data from several well-known sources. There are total of 91,965 samples (examples). It consists of two main groups:

### Original BEANS Tasks

- `esc-50`:  Generic environmental sound classification with 50 labels ([Piczak, 2015](https://dl.acm.org/doi/10.1145/2733373.2806390), License: CC-BY-NC)
- `watkins`: Marine mammal species classification with 31 species ([Sayigh et al., 2016](https://asa.scitation.org/doi/abs/10.1121/2.0000358), free for personal and academic use)
- `cbi`: Bird species classification with 264 labels from the CornellBird Identification competition hosted on Kaggle ([Howard et al., 2020](https://kaggle.com/competitions/birdsong-recognition), License: CC-BY-NC-SA)
- `humbugdb`: Mosquito wingbeat sound classification into 14 species ([Kiskin et al., 2021](https://arxiv.org/abs/2110.07607), License: CC-BY)
- `enabirds`: Bird dawn chorus detection with 34 species ([Chronister et al., 2021](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/ecy.3329), License: CC0)
- `hiceas`: Minke whale detection from the Hawaiian Islands Cetacean and Ecosystem Assessment Survey (HICEAS) ([NOAA, 2022](https://doi.org/10.25921/e12p-gj65), free without restriction)
- `rfcx`: Bird and frog detection from the Rainforest Connection(RFCx) data with 24 species ([LeBien et al., 2020]( https://www.sciencedirect.com/science/article/pii/S1574954120300637.), usage allowed for academic research)
- `gibbons`: Hainan gibbon detection with 3 call type labels ([Dufourq et al., 2021]((https://doi.org/10.1002/rse2.201)), License: CC-BY-NC-SA)

### Newly Added Subsets

- `unseen-species-*`: Unseen species classification with 200 species held out from AnimalSpeak ([Robinson et al., 2024](https://doi.org/10.1109/ICASSP48485.2024.10447250)), with each sub-dataset using common (`cmn`), scientific (`sci`), or taxonomic (`tax`) names
- `unseen-genus-*`: Generalize to unseen genera (`cmn`/`sci`/`tax`)
- `unseen-family-*`: Generalize to unseen families (`cmn`/`sci`/`tax`)
- `lifestage`: Predicting the lifestage of birds across multiple species (e.g., adult, juvenile), curated from [xeno-canto](https://xeno-canto.org/)
- `call-type`: Classifying song vs. call across multiple bird species, curated from [xeno-canto](https://xeno-canto.org/)
- `captioning`: Captioning bioacoustic audio on AnimalSpeak ([Robinson et al., 2024](https://doi.org/10.1109/ICASSP48485.2024.10447250))
- `zf-indv`: Determining whether a recording contains multiplezebra finches, using programmatically generated mixtures (1–4 individuals) ([Elie and Theunissen, 2016](https://doi.org/10.6084/m9.figshare.11905533.v1))

Each sample is labeled with its source dataset and license.

## Usage
```python
import numpy as np
from datasets import load_dataset

ds = load_dataset("EarthSpeciesProject/BEANS-Zero", split="test") 

# see the contents at a glance
print(ds)
```
```python
# get audio for the first sample in the dataset, the 0th index
audio = np.array(ds[0]["audio"])
print(audio.shape)

# get the instruction (prompt / query) for that sample
print(ds[0]["instruction_text"])
# the desired output (should *only* be used for evaluation)
print(ds[0]["output"])

# the component datasets of BEANS-Zero are:
components, dataset_sample_counts = np.unique(ds["dataset_name"], return_counts=True)

# if you want to select a subset of the data, e.g. 'esc50'
idx = np.where(np.array(ds["dataset_name"]) == "esc50")[0]
esc50 = ds.select(idx)
print(esc50)
```

```python
# To stream the dataset instead of downloading it, first
ds = load_dataset("EarthSpeciesProject/BEANS-Zero", split="test", streaming=True)

for i, sample in enumerate(ds):
    # check one sample
    break
print(sample.keys())
```

## Data Fields

The following fields are present in each example:

- **source_dataset** (str): One of the source datasets mentioned above
- **audio** (Sequence[float]): The audio data as a list of floats.
- **id** (str): Sample uuid.
- **created_at** (str): Sample creation datetime in UTC
- **metadata** (str): Each sample can have a different duration (in seconds) and a different sample rate (in Hz). The 'metadata' is a JSON string containing these two fields.
- **file_name** (str): Audio file name
- **instruction** (str): A prompt (a query) corresponding to the audio for your audio-text model with a placeholder for audio tokens. E.g. '<Audio><AudioHere></Audio> What is the scientific name for the focal species in the audio?'
- **instruction_text** (str): Same as **instruction** but without the placeholder for audio tokens.
- **output** (str): The expected output from the model
- **task** (str): The task type e.g. classification / detection / captioning.
- **dataset_name** (str): Names corresponding to the evaluation tasks, e.g. 'esc50' or 'unseen-family-sci'.
- **license** (str): The license of the dataset. For example, 'CC-BY-NC' or 'CC0'.

## Licensing

Due to its composite nature, BEANS-Zero is subject to multiple licenses. Individual samples have the "license" field indicating the specific license for that sample. The dataset is not intended for commercial use, and users should adhere to the licenses of the individual datasets.

## Citation

If you use BEANS-Zero, please cite the following:

```bibtex
@inproceedings{robinson2025naturelm,
  title     = {NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics},
  author    = {David Robinson and Marius Miron and Masato Hagiwara and Olivier Pietquin},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=hJVdwBpWjt}
}
```

## Contact

For questions, comments, or contributions, please contact:
- David Robinson (david at earthspecies dot org)
- Marius Miron (marius at earthspecies dot org)
- Masato Hagiwara (masato at earthspecies dot org)
- Gagan Narula (gagan at earthspecies dot org)
- Milad Alizadeh (milad at earthspecies dot org)
