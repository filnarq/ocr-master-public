# Text recognition
## Preparation
### Dependencies
```
pip install -r requirements.txt
aim init
```
### Models
Copy models to corresponding folder in `CRAFT/pths`
## Usage
### General
```
$ python main.py
Available subcommands: train, test, inference
Fill config.py with values
```
### Dataset generation
`datagen.sh` uses [belval/trdg](https://github.com/Belval/TextRecognitionDataGenerator) Docker image to generate letters dataset