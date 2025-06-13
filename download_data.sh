#!/usr/bin/env bash

wget https://otexts.com/fpppy/data/fpppy_data.zip
unzip fpppy_data.zip -d ./
rm fpppy_data.zip

wget https://otexts.com/fpppy/data/US_change.csv -O ./data/US_change.csv