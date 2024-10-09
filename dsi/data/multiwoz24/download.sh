
#!/bin/bash

# This downloads, preprocesses, and organizes the multiwoz data under /data/multiwoz24/original

# URL to the raw file on GitHub
url="https://github.com/smartyfh/MultiWOZ2.4/archive/refs/heads/main.zip"
output_file="data/MULTIWOZ2.4.zip"

# Create the output directory if it doesn't exist
mkdir -p $output_dir

# Download the zip file
curl -L $url -o $output_file

# Unzip
unzip $output_file -d "data"

rm "data/MULTIWOZ2.4.zip"

cd data/MultiWOZ2.4-main

python create_data.py

cd ../..

mkdir data/multiwoz24
mv data/MultiWOZ2.4-main/data/mwz2.4 data/multiwoz24/original
mv data/MultiWOZ2.4-main data/multiwoz24/MultiWOZ2.4