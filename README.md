# Machine learning of large-scale multimodal brain imaging data reveals neural correlates of handedness 0.0BPublic




## Requirements 
| Environment | Version |
|:---:|:---|
| Python | 3.8.1 |
| Conda | 4.8.2 |

### Setup Development Environment

```
conda create --name mlhand --file .env.conda
```

## Training Models

```
python scripts/train.py --modalities=<MODALITY1,MODALITY2> --artifact-dir=<SOME_PATH>
```

Remarks:
- We use `<...>` denote values needed to be specified. 
- Available modalities can be found at `./resources`.

## Figures, Tables, and Artifacts Mapping 

|ID|Desciption| Notebook|
|:----|:-------|:---|
|Figure 1|Handedness Distribution  | fig-1-handedness-distribution.ipynb |
|Table 3| Main Table with models from different modalities | table-training-statistics.ipynb |
|Table ID-IC-Importance| Table of Top Important ICs | table-IC-performance-statistics.ipynb |
|Figure ID-Importance-Compared| Compare performance of IC models and reference modle | hypothesis-testing.ipynb |
|Test Set Prediction Score | Probablity of each sample being non-righ hander produced by models in nested cross-validation| dev-get-testset-handedness-score.ipynb (on cluster only) |
|IC Importance Statistics| JSON file with statistics of each IC model| table-IC-performance-statistics.ipynb |



## Acknowledgement
...