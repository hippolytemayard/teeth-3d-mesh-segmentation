# 3D Teeth Scan Segmentation 


Implementation of [MeshSegNet: Deep Multi-Scale Mesh Feature Learning for Automated Labeling of Raw Dental Surface from 3D Intraoral Scanners](https://ieeexplore.ieee.org/abstract/document/8984309) [1] for automated tooth segmentation and labeling on raw dental surfaces. 
The dataset used for this project is the challenge 3DTeethSeg22 dataset (associated with MICCAI 2022) [2] and is publicly available.

<img src="./assets/images/mesh-sample.gif" width="400" />

This implementation is based on the paper and the official implementation [![GitHub](https://i.stack.imgur.com/tskMh.png)GitHub](https://github.com/Tai-Hsien/MeshSegNet/)

<img src="./assets/images/meshsegnet_architecture.png" width="800" />


## Data

The dataset used for this project is the challenge 3DTeethSeg22 dataset (associated with MICCAI 2022) [2] and is publicly available. \
It dataset contains a collection of 3D dental surface scans obtained from intraoral scanners. These scans represent the raw dental surfaces of patients and serve as the input data for the segmentation task. The dataset includes a diverse range of dental conditions and variations, capturing different teeth shapes, sizes, and occlusions. A total of 1800 3D intra-oral scans have been collected for 900 patients covering their upper and lower jaws separately.

## Install 

### Library requirements

- Pyenv 2.3.9
- Python 3.10.5    
- Poetry 1.5.1


#### Install Pyenv

```bash
curl https://pyenv.run | bash
```

Then, add pyenv bin directory in your shell configuration file `~/.bashrc` or `~/.zshrc`

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)
```

Finally reload you shell 
```bash
source ~/.bashrc
```

#### Install Poetry


```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then, add poetry's bin directory in your shell configuration file `~/.bashrc` or `~/.zshrc`

```bash
export PATH="/home/ubuntu/.local/bin:$PATH"
```

Reload you shell 
```bash
source ~/.bashrc
```
or if you are using zsh 
```bash
source ~/.zshrc
```

Install your virtual environment
```bash
poetry install
```

## Training

Setup the paths to you training (respectively validation) data directories.

Run the training with the following command line

```bash
poetry run python src/training/train.py 
```

## Monitoring

You can monitor your training using `tensorboard` 

```bash
tensorboard --logdir "path_to_your_log_directory"
```

## TODO

- [x] Training implementation
- [x] Monitoring training
- [ ] Inference function
- [ ] Model evaluation



## References 

[1] Tai-Hsien Wu, Chia-Jung Hsu, Sheng-Hong Huang, et al., "MeshSegNet: Deep Multi-Scale Mesh Feature Learning for Automated Labeling of Raw Dental Surface from 3D Intraoral Scanners," in IEEE Transactions on Medical Imaging, vol. 39, no. 4, pp. 945-956, April 2020. Available at: [Link](https://ieeexplore.ieee.org/abstract/document/8984309)


[2] "3DTeethSeg22_challenge: Dataset for the MICCAI 2022 3D Teeth Segmentation Challenge," GitHub repository, https://github.com/abenhamadou/3DTeethSeg22_challenge.

```
@article{ben20233dteethseg,
title={3DTeethSeg'22: 3D Teeth Scan Segmentation and Labeling Challenge},
author={Achraf Ben-Hamadou and Oussama Smaoui and Ahmed Rekik and Sergi Pujades and Edmond Boyer and Hoyeon Lim and Minchang Kim and Minkyung Lee and Minyoung Chung and Yeong-Gil Shin and Mathieu Leclercq and Lucia Cevidanes and Juan Carlos Prieto and Shaojie Zhuang and Guangshun Wei and Zhiming Cui and Yuanfeng Zhou and Tudor Dascalu and Bulat Ibragimov and Tae-Hoon Yong and Hong-Gi Ahn and Wan Kim and Jae-Hwan Han and Byungsun Choi and Niels van Nistelrooij and Steven Kempers and Shankeeth Vinayahalingam and Julien Strippoli and Aurélien Thollot and Hugo Setbon and Cyril Trosset and Edouard Ladroit},
journal={arXiv preprint arXiv:2305.18277},
year={2023}
}

@article{ben2022teeth3ds,
title={Teeth3DS: a benchmark for teeth segmentation and labeling from intra-oral 3D scans},
author={Ben-Hamadou, Achraf and Smaoui, Oussama and Chaabouni-Chouayakh, Houda and Rekik, Ahmed and Pujades, Sergi and Boyer, Edmond and Strippoli, Julien and Thollot, Aur{\'e}lien and Setbon, Hugo and Trosset, Cyril and others},
journal={arXiv preprint arXiv:2210.06094},
year={2022}
}
```