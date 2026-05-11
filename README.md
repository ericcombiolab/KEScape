# KEScape
This repository contains the source code, model weights of KEScape, as well as the 11 collected deep mutational scanning (DMS) datasets. KEScape is a computational model that predicts the immune escape potential of viral variants relative to a wild-type sequence. Its applications include identifying immune escape hotspots and surveilling emerging viral variants, making it a valuable tool for real-time surveillance during epidemics and pandemics.

# Usage
## Install KEScape
```
git clone https://github.com/ericcombiolab/KEScape.git
conda env create -f kescape_environment.yml
```
To download KEScape model weights, you may download ```esmfold_finetuned.pt``` and a corresponding weight for your task (e.g. hotspots_lineage_surveillance_checkpoints/model_epoch11_no_XBB_lambdal20.05.pt for SARS-CoV-2 lineage surveillance) from the [Hugging Face](https://huggingface.co/charleswang335/KEScape/tree/main). In the Hugging Face repository, ```DMS_checkpoints``` directory contains model weights for various DMS datasets across multiple viral proteins. Each weight was trained exclusively on its respective target protein and was applied in the DMS benchmark. In contrast, ```hotspots_lineage_surveillance_checkpoints``` directory contains a model weight trained on a consolidated dataset that excludes only the SARS-CoV-2 XBB.1.5 spike. This model weight is used for immune escape hotspot identification and lineage surveillance within SARS-CoV-2 spike proteins.


# Citation
If you would like to cite KEScape, please cite the following paper:

Wang, C., Zhang, L.: A structure-informed evolutionary model for predicting viral immune escape and evolution. bioRxiv 2025.07.31.667864; doi: https://doi.org/10.1101/2025.07.31.667864
