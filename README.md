# Fake News Detection Framework with Journalist-in-the-Loop

This repository contains the code for the Fake News Detection Framework implemented as a part of my master thesis under M.Sc. in Intelligent Adaptive Systems. The framework is built and implemented to assist end-users/journalists in their fact-verification process. The complete thesis is present here (add pdf of the report)

## Using the Server

### Hardware Prerequisite
The framework uses two fine-tuned transformer models, BERT and T5. They both are computationally heavy, hence require GPUs for running the server.

Depending on the GPU available, edit the lines in srvFakeNewsDetection.py

```
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5" 

```

### Software Prerequisite
1. The server is a Flask-based server and can be run either on local host or using a public IP for external access. The current port number is set to '5001', which can be changed as required.

2. The library requirement are contained in the requirements.txt file. I suggest to create a python 3 virtual environment to avoid version related issues while installing the requirements. 

Note that, the code is built and tested using the library versions as per the requirements. To ensure the server is running without issues, these versions are mandatory. 

Run the following command to install requirements:

```
pip3 intall -r requirements.txt

```

3. The Spacy uses the English model and requires to be downloaded separately.

Run the following command to download it:

```
python3 -m spacy download en_core_web_l

```

4. The NewsApi library requires APIKEY. Register with them to obtain the key and add it in Utils/WebCrawler.py

```
myKey = "key from newsapi" 

```


### Activate the Server

Once the above prerequites are met, the Flask server can be activated.

Run the following command to activate the server:

```
python3 srvFakeNewsDetection.py

```

The following should be the expected outputs:

<pre>Current Device ID is:0
Class Weights: [1.02800445 0.97348086]
 * Serving Flask app &quot;srvFakeNewsDetection&quot; (lazy loading)
 * Environment: production
<font color="#CC0000">   WARNING: This is a development server. Do not use it in a production deployment.</font>
<font color="#AAAAAA">   Use a production WSGI server instead.</font>
 * Debug mode: off
 * Running on http://0.0.0.0:5001/ (Press CTRL+C to quit)
</pre>


Note: environement depdency 

