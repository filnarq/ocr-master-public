# Text recognition
## Preparation
### Dependencies
```
pip install -r requirements.txt
aim init
```
### Models
Copy models to corresponding folder in `CRAFT/pths`
* [VGG16](https://drive.google.com/open?id=1WzDnOuU_dELMDSaecs1e9uc0aqRBHQX4)
* [Pre-trained model](https://drive.google.com/open?id=1RROAUqBQsydRhGpmUwTT1zu3yWNoVWsO)
* [Finetune model](https://drive.google.com/open?id=1aYa7vv3jOx3TJpyz-CpMbe5pq8ec8dtI)
### Dataset generation
`datagen.sh` uses [belval/trdg](https://github.com/Belval/TextRecognitionDataGenerator) Docker image to generate letters dataset
## Usage
```
$ python main.py
Available subcommands: train, test, inference
Fill config.py with values
```