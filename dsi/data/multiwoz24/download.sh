
#!/bin/bash

# URL to the raw file on GitHub
url="https://github.com/smartyfh/MultiWOZ2.4/raw/main/data/MULTIWOZ2.4.zip"
output_file="MULTIWOZ2.4.zip"
output_dir="data/multiwoz24"

# Create the output directory if it doesn't exist
mkdir -p $output_dir

# Download the zip file
echo "Downloading $output_file..."
curl -L $url -o $output_file

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download successful."

    # Unzip the file into the specified directory
    echo "Unzipping $output_file to $output_dir..."
    unzip $output_file -d $output_dir

    # Check if unzip was successful
    if [ $? -eq 0 ]; then
        echo "Unzipping successful."

        # Clean up: remove the zip file
        echo "Cleaning up: deleting $output_file..."
        rm $output_file

        echo "Cleanup complete."
    else
        echo "Failed to unzip the file."
    fi
else
    echo "Failed to download the file."
fi


