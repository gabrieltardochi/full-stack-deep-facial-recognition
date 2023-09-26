#!/bin/bash

API_URL="http://localhost:8080/api/v1/index"

declare -a PARAMS=(
    '{"image_url": "https://github.com/gabrieltardochi/full-stack-deep-facial-recognition/blob/main/sample-faces/index/elon_musk.jpg?raw=true", "image_format": "jpg", "name": "Elon Musk"}' 
    '{"image_url": "https://github.com/gabrieltardochi/full-stack-deep-facial-recognition/blob/main/sample-faces/index/joe_biden_presidential_portrait.jpg?raw=true", "image_format": "jpg", "name": "Joe Biden"}'
    '{"image_url": "https://github.com/gabrieltardochi/full-stack-deep-facial-recognition/blob/main/sample-faces/index/mark_zuckerberg.jpg?raw=true", "image_format": "jpg", "name": "Mark Zuckerberg"}'
)

# Loop through the parameters and send POST requests
for param in "${PARAMS[@]}"; do
    curl -X POST -d "$param" -H "Content-Type: application/json" "$API_URL" &
    sleep 5
done

# Wait for all background jobs to finish
wait