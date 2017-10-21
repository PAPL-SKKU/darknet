#!/bin/sh

# ./darknet classifier predict cfg/imagenet1k.data cfg/alexnet.cfg alexnet.weights data/dog.jpg
./darknet classifier predict cfg/imagenet1k.data cfg/extraction.cfg extraction.weights data/giraffe.jpg
# ./darknet classifier predict cfg/imagenet1k.data cfg/alexnet.cfg alexnet.weights data/dog.jpg
# ./darknet classifier valid cfg/imagenet1k.data cfg/alexnet.cfg alexnet.weights
