# PRE-DFKD
[![built with Python3.6](https://img.shields.io/badge/build%20with-python%203.6-red.svg)](https://www.python.org/)
[![built with PyTorch1.4](https://img.shields.io/badge/build%20with-pytorch%201.4-brightgreen.svg)](https://pytorch.org/)

Official implementation of the paper Robust and Resource Efficient Data-Free Knowledge Distillation by Generative Pseudo Replay (AAAI'22)

Authors: Kuluhan Binici, Shivam Aggarwal, Pham Nam Trung, Tulika Mitra, Karianto Leman

## Project Structure
The project is structured as follows:

- [`networks/`](networks): This folder contains PyTorch neural network descriptions for teacher/ student models.
- [`utils/`](utils): Collection of auxiliary scripts that help the main functionality.
- [`distillation.py`](distillation.py): Script that contains core functions to be used in distillation.
- [`main.py`](main.py): Main PRE-DFKD script 
- [`student-eval.py`](student-eval.py): Script to evaluate student models.
- [`README.md`](README.md): This instructions file.
- [`LICENSE`](LICENSE): CC BY 4.0 Licence file.

## Running the code

First create a cache directory in the parent directory of this project with two sub-folders: [`data/`] and [`models/`]
```
+-- cache
    +-- data
    +-- models
+-- PRE-DFKD
    +-- ...
```

The datasets will be downloaded to the [`cache/data`] folder

Then, obtain teacher models to be distilled and save them in [`cache/models`] folder

### Download pre-trained teachers

Download link: https://drive.google.com/drive/folders/1fTUl8Igs5gEbWrdrwZd_22YEhl9ZO7rP?usp=sharing

### run PRE-DFKD (dataset: CIFAR100, teacher: ResNet34, student: ResNet18)

```bash
python cifar10.sh
```

## Citation

Please consider citing our work if you make use of it
```
@article{binici2022robust,
  title={Robust and Resource-Efficient Data-Free Knowledge Distillation by Generative Pseudo Replay},
  author={Binici, Kuluhan and Aggarwal, Shivam and Pham, Nam Trung and Leman, Karianto and Mitra, Tulika},
  journal={arXiv preprint arXiv:2201.03019},
  year={2022}
}
```

