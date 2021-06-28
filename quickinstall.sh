echo Running scripts for quick installation of the Fake News Detection Framework

wget -nd -r --no-parent --reject "index.html*" http://ltdata1.informatik.uni-hamburg.de/factverify/TrainedModels/ -P TrainedModels

python3 -m venv srvEnv

source srvEnv/bin/activate

pip3 install -r requirements.txt

python3 -m spacy download en_core_web_l    