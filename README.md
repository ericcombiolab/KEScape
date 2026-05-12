# KEScape
This repository contains the source code, model weights of KEScape, as well as the 11 collected deep mutational scanning (DMS) datasets. KEScape is a computational model that predicts the immune escape potential of viral variants relative to a wild-type sequence. Its applications include identifying immune escape hotspots and surveilling emerging viral variants, making it a valuable tool for real-time surveillance during epidemics and pandemics.

# Usage
## Install KEScape
```
git clone https://github.com/ericcombiolab/KEScape.git
conda env create -f kescape_environment.yml
```
To download KEScape model weights, you may download ```esmfold_finetuned.pt``` and a corresponding weight for your task (e.g. hotspots_lineage_surveillance_checkpoints/model_epoch11_no_XBB_lambdal20.05.pt for SARS-CoV-2 lineage surveillance) from the [Hugging Face](https://huggingface.co/charleswang335/KEScape/tree/main). The ```esmfold_finetuned.pt``` should be put into the same directory with the training and inference scripts. In the Hugging Face repository, ```DMS_checkpoints``` directory contains model weights for various DMS datasets across multiple viral proteins. Each weight was trained exclusively on its respective target protein and was applied in the DMS benchmark. In contrast, ```hotspots_lineage_surveillance_checkpoints``` directory contains a model weight trained on a consolidated dataset that excludes only the SARS-CoV-2 XBB.1.5 spike. This model weight is used for immune escape hotspot identification and lineage surveillance within SARS-CoV-2 spike proteins.

## Train KEScape
The script for training KEScape is in ```DDP_training.py```. Before training KEScape, you need to set the user-defined arguments first. They are listed and explained as follows:
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
python DDP_training.py --train_data_path $TRAIN_DATA_PATH --val_data_path $VAL_DATA_PATH --ratio $RATIO --batch_size $BATCH_SIZE --checkpoint_prefix $CHECKPOINT_PREFIX --train_epochs $TRAIN_EPOCHS
```

## KEScape inference
The script for using KEScape for inference is in ```inference.py```. Before running inference, you need to set the user-defined arguments first. They are listed and explained as follows:
```
batch_size: the batch size for inference
data_path: the path to the test dataset
checkpoint_path: the path to the saved model during the training stage
output_path: the path to the output file to save the predicted scores
```
After setting these inputs, you can run inference by the following:
```
python inference.py --data_path $DATA_PATH --checkpoint_path $CHECKPOINT_PATH --output_path $OUTPUT_PATH --batch_size $BATCH_SIZE
```

## File formats required by KEScape
For training, KEScape requires three types of inputs: wild-type sequences, mutant sequences, and labels. They should be stored in a csv file as follows:
```
raw_seq,mut_seq,label
...,...,...
```
For inference, KEScape requires two types of inputs: wild-type sequences, mutant sequences. They should be stored in a csv file as follows:
```
raw_seq,mut_seq
...,...
```
We provide a sample dataset for KEScape in ```data/sample_demo/```. Note that there are additional columns in the sample data, but they are not a must for KEScape.

# DMS datasets
In total, 11 DMS datasets were collected. You may find their sources in the supplementary file of the KEScape paper. The raw files of these datasets are stored in ```data/raw_DMS```. The processed files are stored in ```data/processed_DMS```, which have a uniform format as follows:
```
pos,wildtype,mutant,label
...,...,...,...
```

# Citation
If you would like to cite KEScape, please cite the following paper:

Wang, C., Zhang, L.: A structure-informed evolutionary model for predicting viral immune escape and evolution. bioRxiv 2025.07.31.667864; doi: https://doi.org/10.1101/2025.07.31.667864
