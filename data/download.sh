#!/bin/bash

# Download soil classification dataset
echo "Downloading soil classification dataset..."

# Download dataset (example using wget)
wget -O soil_dataset.zip "https://example.com/soil_dataset.zip"

# Extract dataset
unzip soil_dataset.zip -d data/

# Clean up
rm soil_dataset.zip

echo "Dataset download complete!"
