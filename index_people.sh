#!/bin/bash

API_URL="http://localhost:8080/api/v1/index"

PARAMS=(
    "'{ "image_url": , "image_format": , "name": }'" 
    "param2=value2" "param3=value3")

# Loop through the parameters and send POST requests
for param in "${PARAMS[@]}"; do
    curl -X POST -d "$param" -H "Content-Type: application/json" "$API_URL" 
    wait $!
    sleep 5
done