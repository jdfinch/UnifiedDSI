#!/bin/bash

# This downloads, preprocesses, and organizes the DSTC8 Schema-Guided Dialogue dataset under /data/dstc8/original

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

# Navigate into the unzipped directory
cd data/dstc8-schema-guided-dialogue-master

# If there's a preprocessing script, you can run it here
# For example:
# python preprocess_data.py

# Navigate back to the root directory
cd ../..

# Create a new directory to organize the dataset
mkdir -p data/dstc8/original
mkdir -p data/dstc8/SGD

# Move the dataset to the new directory
mv data/dstc8-schema-guided-dialogue-master/dev data/dstc8/original
mv data/dstc8-schema-guided-dialogue-master/test data/dstc8/original
mv data/dstc8-schema-guided-dialogue-master/train data/dstc8/original

# Optionally, move the entire unzipped directory for reference
mv data/dstc8-schema-guided-dialogue-master data/dstc8/SGD

mv data/dstc8 data/sgd
