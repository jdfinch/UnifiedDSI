#!/bin/bash

data_url="https://raw.githubusercontent.com/emorynlp/Diverse0ShotTracking/refs/heads/main/data/dsg5k/train"
slots_url="$data_url/slot.csv"
slot_values_url="$data_url/slot_value.csv"
turns_url="$data_url/turn.csv"

mkdir -p data/d0t/original

echo "Downloading slots.csv..."
curl -o data/d0t/original/slot.csv "$slots_url"

echo "Downloading slot_value.csv..."
curl -o data/d0t/original/slot_value.csv "$slot_values_url"

echo "Downloading turn.csv..."
curl -o data/d0t/original/turn.csv "$turns_url"

echo "Download complete!"

