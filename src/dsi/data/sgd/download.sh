#!/bin/bash

# This downloads, preprocesses, and organizes the DSTC8 Schema-Guided Dialogue dataset under /data/sgd/original

# URL to the raw file on GitHub
url="https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/archive/refs/heads/master.zip"
output_file="data/dstc8-schema-guided-dialogue.zip"

# Create the output directory if it doesn't exist
mkdir -p data

# Download the zip file
curl -L $url -o $output_file

# Unzip the downloaded file into the 'data' directory
unzip $output_file -d "data"

# Remove the zip file to save space
rm $output_file

# Create a new directory to organize the dataset
mkdir -p data/sgd/original

# Move the dataset to the new directory
mv data/dstc8-schema-guided-dialogue-master/dev data/sgd/original/dev
mv data/dstc8-schema-guided-dialogue-master/test data/sgd/original/test
mv data/dstc8-schema-guided-dialogue-master/train data/sgd/original/train

# Optionally, move the entire unzipped directory for reference
mv data/dstc8-schema-guided-dialogue-master data/sgd/meta
