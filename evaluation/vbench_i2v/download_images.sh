#!/bin/bash
# Download VBench-I2V image suite and metadata.
#
# Usage:
#   bash evaluation/vbench_i2v/download_images.sh
#
# Prerequisites:
#   pip install gdown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

mkdir -p "$DATA_DIR"

echo "Downloading i2v-bench-info.json ..."
gdown --id 1zmWs_m_A4q6YgTZwIZ230jW0ttknlGJA --output "$DATA_DIR/i2v-bench-info.json"

echo "Downloading cropped images ..."
gdown --id 1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ --output "$DATA_DIR/crop.zip"

echo "Downloading vbench2_i2v_full_info.json ..."
FULL_INFO_URL="https://raw.githubusercontent.com/Vchitect/VBench/master/vbench2_beta_i2v/vbench2_i2v_full_info.json"
wget -q -O "$DATA_DIR/vbench2_i2v_full_info.json" "$FULL_INFO_URL"

echo "Unzipping ..."
unzip -qo "$DATA_DIR/crop.zip" -d "$DATA_DIR"
rm -f "$DATA_DIR/crop.zip"

echo "Done! Images at: $DATA_DIR/crop/"
echo "Available aspect ratios:"
ls "$DATA_DIR/crop/"
