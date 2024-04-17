## About this script

This project is written in nodejs. It uses the [face-api](https://github.com/vladmandic/face-api) node library in combination with tensorflow to find a matching face in a database of images.

## The two scripts

This project is broken down in 2 scripts:

### generateFaceDescriptors

This script is responsible for generating the face descriptors that are used in the faceMatcher script to find a matching face. It is necessary to pre-generate these descriptors because the time it takes to calculate the descriptors of 100k images would be multiple hours, so it can't be done at runtime. The calculated descriptors are stored in assets/faceDescriptors.

### faceMatcher

This is the script that takes an image of a face as input, and outputs a label name that corresponds with one of the images that were used to pre-calculate the face descriptors.
