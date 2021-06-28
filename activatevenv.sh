#!/bin/sh

python3 -m venv srvEnv
source ./srvEnv/bin/activate

pip3 install -r requirements.txt

#python3 -m spacy download en_core_web_lg
