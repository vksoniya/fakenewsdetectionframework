#!/bin/sh

python3 -m venv srvEnv
source ./srvEnv/bin/activate

#pip3 install -r requirements.txt

#python3 -m spacy download en_core_web_lg

# Uncomment this code in case by default your environment is not using python 3.8 
#conda create -n srvEnv python=3.8

pip3 install -r requirements.txt
python3 -m spacy download en_core_web_lg
python3 srvFakeNewsDetection.py
