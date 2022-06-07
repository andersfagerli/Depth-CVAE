# Depth-CVAE
PyTorch implementation of a conditional variational autoencoder for predicting depth from images.

![](demo/movie.gif)

## Requirements
### PyTorch
Choose your relevent PyTorch version here https://pytorch.org/get-started/locally/, by choosing correct system, pip/conda, GPU/CPU only. E.g for Linux using pip with no GPU, this would be

```
pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
### Additional requirements
Download the additional requirements by
```
pip3 install -r requirements.txt
```

## Datasets
### VAROS
Download the VAROS dataset from https://zenodo.org/record/5567209#.YcEgMVPMJhE, and place it in the datasets/ folder.

Example of a valid folder structure:
```
VAROS_ROOT
|__ 2021-08-17_SEQ1
    |__ train
        |__ vehicle0
            |__cam0
                |__A
                |__B
                |__C
                |__D
|__ ...
```

## Configuration
Configuration files are located in configs/, where you can set parameters, location of trained model, demo images etc.

## Train
After downloading and placing the datasets correctly, do e.g.
```
python3 train.py configs/varos.yaml
```
to train on the VAROS dataset.

## Testing
After having trained a model, do e.g.
```
python3 demo.py configs/varos.yaml
```
to test on a set of demo images located in demo/

## Acknowledgements
Implementation is based on https://github.com/lufficc/SSD.
