# KEScape
This repository contains the source code of KEScape. KEScape is a computational model that predicts the immune escape potential of viral variants relative to a wild-type virus. Its applications include identifying immune escape hotspots and surveilling emerging viral variants.

# Usage
## Install KEScape
```
git clone https://github.com/ericcombiolab/KEScape.git
conda env create -f kescape_environment.yml
```
To download KEScape model weights, you may download ```esmfold_finetuned.pt``` and ```KEScape.pt``` from [Hugging Face](https://huggingface.co/charleswang335/KEScape/tree/main). Both files should be placed in the ```checkpoint/``` directory. ```KEScape.pt``` was trained on a pooled dataset consisting of nine DMS datasets (H1N1 HA, H3N2 HA, H5N1 HA, SARS-CoV-2 spike RBD, HIV envelope, Zika envelope, rabies glycoprotein, Nipah receptor-binding protein, and Lassa glycoprotein).

## Train KEScape
The script for training KEScape is in ```DDP_train.py```. Before training KEScape, you need to set the user-defined arguments first. They are listed and explained as follows:
```
ratio: the ratio of positive samples to negative samples in the training dataset
train_data_path: the path to the training dataset
val_data_path: the path to the validation dataset
checkpoint_prefix: the prefix for the saved model checkpoints, the saved model will be named as {model_prefix}_epoch{epoch}_lambdal2{lambda_l2}.pt
batch_size: the batch size for training and evaluation
train_epochs: the number of epochs for training
```
After setting these inputs, you can train KEScape by running:
```
python DDP_train.py --train_data_path $TRAIN_DATA_PATH --val_data_path $VAL_DATA_PATH --ratio $RATIO --batch_size $BATCH_SIZE --checkpoint_prefix $CHECKPOINT_PREFIX --train_epochs $TRAIN_EPOCHS
```

## KEScape inference
The inference script is ```inference.py```. It accepts the following arguments:
```
input: the path to the input CSV file
checkpoint: the path to the KEScape checkpoint (default: checkpoint/KEScape.pt)
output: the path to the output CSV file
batch_size: the batch size for inference (default: 1)
```
Run inference with:
```
python inference.py --input sample_data/sample.csv --output outputs/sample_predictions.csv --batch_size 1
```
The output contains the input columns and an additional ```prediction``` column.

## File formats required by KEScape
For training, KEScape requires wild-type sequences, mutant sequences, and labels in a CSV file:
```
wildtype_sequence,mutant_sequence,label
...,...,...
```
For inference, KEScape requires only wild-type and mutant sequences:
```
wildtype_sequence,mutant_sequence
...,...
```
A sample inference dataset containing 10 sequence pairs is provided in ```sample_data/sample.csv```.

# DMS datasets
KEScape was developed and evaluated using 13 DMS datasets. These datasets are available from the following sources:

| DMS dataset | Source |
|---|---|
| SARS-CoV-2 spike | [EVEscape](https://github.com/OATML-Markslab/EVEscape) |
| SARS-CoV-2 XBB.1.5 spike | [Dadonaite et al., Nature](https://www.nature.com/articles/s41586-024-07636-1) |
| H1N1 hemagglutinin | [EVEscape](https://github.com/OATML-Markslab/EVEscape) |
| HIV envelope | [EVEscape](https://github.com/OATML-Markslab/EVEscape) |
| H5N1 hemagglutinin | [Flu_H5_American-Wigeon_South-Carolina_2021-H5N1_DMS](https://github.com/dms-vep/Flu_H5_American-Wigeon_South-Carolina_2021-H5N1_DMS) |
| Zika envelope | [ZIKV_DMS_with_EvansLab](https://github.com/jbloomlab/ZIKV_DMS_with_EvansLab) |
| Nipah receptor-binding protein | [Nipah_Malaysia_RBP_DMS](https://github.com/dms-vep/Nipah_Malaysia_RBP_DMS) |
| Lassa glycoprotein | [LASV_Josiah_GP_DMS](https://github.com/dms-vep/LASV_Josiah_GP_DMS) |
| H3N2 hemagglutinin | [flu_h3_hk19_dms](https://github.com/dms-vep/flu_h3_hk19_dms) |
| H3N2 neuraminidase | [Lei et al., Cell Reports](https://www.cell.com/cell-reports/fulltext/S2211-1247(23)01368-2) |
| Rabies glycoprotein | [RABV_Pasteur_G_DMS](https://github.com/dms-vep/RABV_Pasteur_G_DMS) |
| SARS-CoV-2 BA.1 spike | [omicron_ab_landscape](https://github.com/desai-lab/omicron_ab_landscape) |
| SARS-CoV-2 BA.2 spike | [SARS-CoV-2_Omicron_BA.2_spike_DMS_sera_and_mAbs](https://github.com/dms-vep/SARS-CoV-2_Omicron_BA.2_spike_DMS_sera_and_mAbs) |

# Citation
If you would like to cite KEScape, please cite the following paper:

Wang, C., Zhang, L.: A structure-informed evolutionary model for predicting viral immune escape and evolution. bioRxiv 2025.07.31.667864; doi: https://doi.org/10.1101/2025.07.31.667864
