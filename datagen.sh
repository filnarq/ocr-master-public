#!/bin/sh

DATASET_NAME=example

# # Generate
# mkdir "data/${DATASET_NAME}"
# docker run -v "./data/${DATASET_NAME}/:/app/out/" -v ./data/dictionary:/app/dictionary -t belval/trdg:latest trdg -t 4 -b 3 -tc '#000000,#FFFFFF' -i /app/dictionary -k 30 -rk -bl 1 -rbl -fi -m 0 -wd 0 -c 62000

cd "data/${DATASET_NAME}"
echo 'Creating classdirs...'
for let in $(ls | cut -d'_' -f1 | sort | uniq); do mkdir $let; done;
echo 'Moving to classdirs...'
for let in $(ls | grep .jpg | cut -d'_' -f1 | sort | uniq); do mv "${let}_*" "${let}/"; done;
echo 'Creating test data...'
mkdir "../${DATASET_NAME}_test"
for folder in $(ls); do mkdir "../${DATASET_NAME}_test/${folder}" && mv "${folder}/$(ls $folder | head -n1)" "../${DATASET_NAME}_test/${folder}/"; done;
