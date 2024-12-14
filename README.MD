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
([backup link](https://drive.google.com/file/d/19D9TRB3eIEMFDRyHQ3cDFbRo9jgik33Q/view?usp=drive_link))
* [Pre-trained model](https://drive.google.com/open?id=1RROAUqBQsydRhGpmUwTT1zu3yWNoVWsO)
([backup link](https://drive.google.com/file/d/1of-eNW5T-MaihVTLqZqbF__3lRa5AxK5/view?usp=drive_link))
* [Finetune model](https://drive.google.com/open?id=1aYa7vv3jOx3TJpyz-CpMbe5pq8ec8dtI)
([backup link](https://drive.google.com/file/d/1PimtbwsHkaRRAMRb1E1GotNILArP8ApL/view?usp=drive_link))
### Dataset generation
`datagen.sh` uses [belval/trdg](https://github.com/Belval/TextRecognitionDataGenerator) Docker image to generate letters dataset
## Usage
### General
```
$ python main.py
Available subcommands: train, test, inference
Fill config.py with values
```
### Training monitoring
For training process monitoring [AimStack](https://aimstack.io/) is used
```
$ aim up
Running Aim UI on repo `<Repo#-7217583351485859554 path=.aim read_only=None>`
Open http://127.0.0.1:43800
Press Ctrl+C to exit
```