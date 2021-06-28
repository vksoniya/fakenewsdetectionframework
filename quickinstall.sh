#!/bin/sh

echo Running scripts for downloading fine-tuned Models of the Fake News Detection Framework

wget -nd -r --no-parent --reject "index.html*" http://ltdata1.informatik.uni-hamburg.de/factverify/TrainedModels/ -P TrainedModels

