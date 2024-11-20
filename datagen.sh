#!/bin/sh

# mkdir data/skewed
# docker run -v ./data/skewed/:/app/out/ -v ./data/dictionary:/app/dictionary -t belval/trdg:latest trdg -t 4 -b 3 -tc '#000000,#FFFFFF' -i /app/dictionary -k 30 -rk -bl 1 -rbl -fi -m 0 -wd 0 -c 62000

cd data/skewed
echo 'Creating classdirs...'
for let in $(ls | cut -d'_' -f1 | sort | uniq); do mkdir $let; done;
echo 'Moving to classdirs...'
for file in $(ls | grep .jpg); do mv $file "$(echo $file | cut -d'_' -f1)/"; done;
echo 'Creating test data...'
mkdir ../skewed_test
for folder in $(ls); do mkdir ../skewed_test/$folder && mv "${folder}/$(ls $folder | head -n1)" ../skewed_test/$folder/; done;
