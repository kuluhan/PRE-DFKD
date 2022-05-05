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

First, obtain teacher models to be distilled

### Download pre-trained teachers

Download link: https://drive.google.com/drive/folders/1fTUl8Igs5gEbWrdrwZd_22YEhl9ZO7rP?usp=sharing

### run PRE-DFKD to transfer knowledge from ResNet34 teacher trained on cifar100 dataset to ResNet18 student

```python
python cifar10.sh
```
