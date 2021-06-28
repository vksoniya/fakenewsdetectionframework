from __future__ import unicode_literals
from flask import Flask, render_template, url_for, request, jsonify, make_response, Response, send_file, session
import os
import os.path
import csv
import glob
import pandas as pd
import datetime
import numpy as np
import transformers
import torch
import torch.nn as nn
import re
from transformers import pipeline
from transformers import AutoModel, BertTokenizerFast
from sklearn.utils.class_weight import compute_class_weight
import sklearn as sk
from tqdm.auto import tqdm, trange

from Utils.KeywordExtractor import extract_keywords
from Utils.WebCrawler import crawl_web
from Utils.TestDataLoader import create_test_dataloader
from Utils.VeracityExplanator import generateExplantion, extractEvidence
from Utils.Justification import generateJustificationText
from Utils.SimilarClaimExtractor import extractCosineSimilarityText

#from Utils.T5Summarizer import getsummaryusingT5

app = Flask(__name__)
app.secret_key = "SECRETKEY"

labels_path = "TrainedModels/train_labels.csv"
model_path = "TrainedModels/ds_weights_30_lr-1e05.pt"

#model_path = "/srv/home/8vijayak/finalDisInfoCheckModule/TrainedModels/ds_weights_30_lr-1e05.pt"

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Current Device ID is:" + str(torch.cuda.current_device()))
    torch.cuda.empty_cache()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="5"  # specify which GPU(s) to be used
else:
    device = torch.device('cpu')
    print("We are running on CPU")

#device = torch.device('cuda')

ButtonPressed = 0

folder = "PipelineOutputs"
files_path = os.path.join(folder, '*')

def generateClaimID(ip_address):
    ext = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    claimID = "_claimID_" + str(ext) + "_" + str(ip_address)

    return claimID

def getTrainLabels():
    labels_df = pd.read_csv(labels_path)
    labels_df.columns = ["index"]
    train_labels = labels_df['index'].values.tolist() 

    return train_labels

class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert 
      
        # relu activation function
        self.relu =  nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)
        #define the forward pass
    def forward(self, sent_id, mask):
        #pass the inputs to the model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        # x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)
        return x

bert = AutoModel.from_pretrained('bert-base-uncased')
# freeze all the parameters of bert, no finetuning of bert
for param in bert.parameters():
    param.requires_grad = False

model = BERT_Arch(bert)
# push the model to GPU
#model.cuda(3)
#torch.cuda.empty_cache()
model = model.to(device)

train_labels = getTrainLabels()
#compute the class weights
class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
print("Class Weights:",class_weights)
# converting list of class weights to a tensor
weights= torch.tensor(class_weights,dtype=torch.float)
# push to GPU
weights = weights.to(device)
#weights= weights.cuda(3)
#torch.cuda.empty_cache()

# define the loss function
cross_entropy  = nn.NLLLoss(weight=weights) 

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


#Prediction function
def predict(model, test_dataloader, verbose=False):
    #print("\nPredicting...")
    model.eval()
    total_loss, total_accuracy = 0, 0
    total_preds = []
    y_true = []
    metrics = {}
    step=0
    
    for batch in tqdm(test_dataloader, desc="predicting..."):
        step=step+1
        batch = [t.to(device) for t in batch]
        #batch = [t.cuda(3) for t in batch]
        sent_id, mask, labels = batch
        y_true.append(labels.detach().cpu().numpy())
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds,labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
    avg_loss = total_loss / len(test_dataloader) 
    total_preds  = np.concatenate(total_preds, axis=0).argmax(axis=1)
    y_true  = np.concatenate(y_true, axis=0)
    
    metrics['loss'] = avg_loss
    metrics['accuracy'] = sk.metrics.accuracy_score(y_true, total_preds)
    metrics['f1'] = sk.metrics.f1_score(y_true, total_preds)
    
    return metrics, total_preds

#function to save all the outputs
def _persist(outputFilename,_claim_id, df_prediction_output, df_evidence):
    file_exists = os.path.isfile(outputFilename)
    if file_exists:
        with open(outputFilename, 'a+') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow([_claim_id, df_prediction_output['Input Claim'], df_prediction_output['Input Keywords'], df_prediction_output['Score Metrics'], df_prediction_output['Accuracy'], df_prediction_output['Generated Justification'], df_prediction_output['Justification Keywords'], df_evidence ])
    else:
        with open(outputFilename, 'a+') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['Claim ID','Input Claim', 'Input Claim Keywords', 'Score Metrics', 'Accuracy', 'Generated Justification', 'Justification Keywords', 'Evidence Sources'])
            tsv_writer.writerow([_claim_id, df_prediction_output['Input Claim'], df_prediction_output['Input Keywords'], df_prediction_output['Score Metrics'], df_prediction_output['Accuracy'], df_prediction_output['Generated Justification'], df_prediction_output['Justification Keywords'], df_evidence ])

            
###Mandate code: for loading
@app.route('/')
def index():
    print("Inside Index function")
    return render_template('index.html')

#For the results page
@app.route('/result', methods=['POST'])
def get_result():
    
    _input_claim = ""
    _extracted_keywords_input_claim = ""
    _classifier_result = ""
    _generated_justification = ""
    _extracted_keywords_generated_justification = ""
    
    try:
        _ip_address = ""
        #Requester IP for recording response
        _ip_address = request.remote_addr
        if (_ip_address == ''):
            _ip_address = "Unknown"
        print("Requester IP: " + str(_ip_address))
        _claim_id = generateClaimID(_ip_address)
        session["CLAIMID"] = _claim_id
        session["IPADDRESS"] = _ip_address
        
        outputFilename= "PipelineOutputs/output_" + _ip_address + ".tsv" 
        session['OUTPUTFILENAME'] = outputFilename
        file_exists = os.path.isfile(outputFilename)
        _feedback_text = ""

        # Step 1: Get input claim
        _input_claim = request.form['inputclaimtext']
        print("Input Claim:" + _input_claim)

        df_prediction_output = {} #dict containting all the outputs
        df_prediction_output = {'Input Claim': _input_claim,
                                'Input Keywords': "",
                                'Score Metrics': "",
                                'Accuracy': "",
                                'Generated Justification': "",
                                'Justification Keywords': ""}
        df_evidence = {} #CHANGE: this needs to be changed and maybe not required

        # Step 2: Classify using Model
        # Step 2a: Extract keywords
        _input_keywords = extract_keywords(_input_claim)

        if (len(_input_keywords) > 0):
            
            df_prediction_output = {'Input Claim': _input_claim,
                                'Input Keywords': _input_keywords,
                                'Score Metrics': "",
                                'Accuracy': "",
                                'Generated Justification': "",
                                'Justification Keywords': ""}

            _extracted_keywords_input_claim = ""
            
            for keyword in _input_keywords:
                print(keyword[0])
                _extracted_keywords_input_claim = _extracted_keywords_input_claim + keyword[0] + ", "
            _extracted_keywords_input_claim = str(_extracted_keywords_input_claim[:-2])
            

            # Step 2b: Crawl Web
            _df_crawled_articles = crawl_web(_input_keywords)
            _df_crawled_articles.to_csv("Datasets/crawled_articles.tsv", sep='\t', index=False)
            
            if(len(_df_crawled_articles)) > 0:
                
                #create prediction set with top 10 similar articles to the input claim
                ### Added similarity code
                print("Total crawled aticles is:" + str(len(_df_crawled_articles['Crawled Article Text'])))
                _df_predictionset = extractCosineSimilarityText(_input_claim, _df_crawled_articles)
                print("Top 10 similar articles are:" + str(len(_df_predictionset['Crawled Article Text'])))
                ### Added similarity code

                # Step 2c: Test Data Loader 
                test_dataloader, final_labels_test = create_test_dataloader(_df_predictionset, _input_claim)

                if(len(test_dataloader)>0):
                    # Step 2d: Predict using model
                    final_score_metrics, final_t_predictions = predict(model, test_dataloader, verbose=False)
                    print("My Model: Test Set: User Input Claim:" + str(final_score_metrics))
                    #Check prediction accuracy
                    final_acc_score = sk.metrics.accuracy_score(final_labels_test, final_t_predictions)
                    print("Model Accuracy: " + str(final_acc_score))
                    f_prediction_result = final_acc_score * 100
                    _classifier_result = "Assuming the input claim is true, the prediction accuracy is: " + str("{:.2f}".format(round(f_prediction_result, 2))) + "%"

                    #Step 3: Generate Justification from Input Claim
                                        
                    ### Added code for BART Summarizer
                    #generated_justification_text = generateJustificationText("BART", df_similar_text)
                    
                    #_tmp_df = _df_predictionset.iloc[:3]
                    
                    #generated_justification_text = getsummaryusingT5(_tmp_df)
                    generated_justification_text = generateJustificationText("TFIDF", _df_predictionset, _input_keywords)
                    #generated_justification_text = generateJustificationText("TFIDF", _df_crawled_articles, _input_keywords)
                    
                    
                    print("Generated justification: " +  generated_justification_text)
                    if(len(generated_justification_text) > 0):
                        _generated_justification = generated_justification_text

                        df_prediction_output = {'Input Claim': _input_claim,
                                                    'Input Keywords': _input_keywords,
                                                    'Score Metrics': final_score_metrics,
                                                    'Accuracy': final_acc_score,
                                                    'Generated Justification': _generated_justification,
                                                    'Justification Keywords': ""}

                        #Step 3a
                        _keywords_justification = extract_keywords(generated_justification_text)
                        _extracted_keywords_generated_justification = ""
                        if len(_keywords_justification) > 0:
                            for keyword in _keywords_justification:
                                print(keyword[0])
                                _extracted_keywords_generated_justification = _extracted_keywords_generated_justification + keyword[0] + ", "
                            _extracted_keywords_generated_justification = str(_extracted_keywords_generated_justification[:-2])

                            #Save results before rendering the results
                            df_prediction_output = {'Input Claim': _input_claim,
                                                    'Input Keywords': _input_keywords,
                                                    'Score Metrics': final_score_metrics,
                                                    'Accuracy': final_acc_score,
                                                    'Generated Justification': _generated_justification,
                                                    'Justification Keywords': _extracted_keywords_generated_justification}

                            #Step 4: Extract evidence from internet
                            #df_evidence = extractEvidence(_input_claim, _keywords_justification)
                            
                            df_evidence = _df_predictionset[['Similarity Score', 'Crawled Article Title', 'Crawled Article Link']]
                            
                            if(len(df_evidence) > 0):
                                #df_evidence = df_evidence.drop(df_evidence.columns[1], axis=1) 
                                df_evidence.columns = ["Similarity Score", "Similar Claim Title", "Source"]
                                #df_evidence.drop_duplicates(subset ="Similarity Score", keep = False, inplace = True)
                                df_evidence.drop_duplicates(subset ="Similar Claim Title", keep = False, inplace = True)
                                df_evidence.reset_index(drop=True, inplace=True)

                                _persist(outputFilename,_claim_id, df_prediction_output, df_evidence)

                                return render_template('result.html', input_claim=_input_claim, extracted_keywords_input_claim=_extracted_keywords_input_claim, 
                                        classifier_result=_classifier_result, generated_justification = _generated_justification, extracted_keywords_generated_justification = _extracted_keywords_generated_justification, similarity_claims = "", table = df_evidence.to_html (header = 'true'), column_names=df_evidence.columns.values, row_data=list(df_evidence.values.tolist()), zip=zip)
                            #Step 4:
                            else:
                                df_evidence = {}
                                _persist(outputFilename,_claim_id, df_prediction_output, df_evidence)
                                
                                return render_template('result.html', input_claim=_input_claim, extracted_keywords_input_claim=_extracted_keywords_input_claim, 
                                        classifier_result=_classifier_result, generated_justification = _generated_justification, extracted_keywords_generated_justification = _extracted_keywords_generated_justification, similarity_claims = "", table = df_evidence.to_html (header = 'true'), column_names=df_evidence.columns.values, row_data=list(df_evidence.values.tolist()), zip=zip)
                        #Step 3a   
                        else:
                            _extracted_keywords_generated_justification = "Could not extract relevant keywords"
                            df_evidence = {}

                            _persist(outputFilename,_claim_id, df_prediction_output, df_evidence)
                    #Step 3        
                    else:
                        _generated_justification = "Could not generate relevant justification"
                        _extracted_keywords_generated_justification = ""
                        df_evidence = {}

                        _persist(outputFilename,_claim_id, df_prediction_output, df_evidence)
                #Step 2c
                else:
                    _classifier_result = "Could not classify invalid claim"
                    _generated_justification = ""
                    _extracted_keywords_generated_justification = ""
                    df_evidence = {}

                    _persist(outputFilename, _claim_id, df_prediction_output, df_evidence)
            #Step 2b
            else: 
                _classifier_result = "Could not crawl the web/ Could not classify invalid claim"
                _generated_justification = ""
                _extracted_keywords_generated_justification = ""
                df_evidence = {}

                _persist(outputFilename, _claim_id, df_prediction_output, df_evidence)
        #Step 2a
        else:
            #When keywords from input text could not be extracted
            _extracted_keywords_input_claim = "Could not extract valid keywords from input"
            _classifier_result = ""
            _generated_justification = ""
            _extracted_keywords_generated_justification = ""
            df_evidence = {}

            _persist(outputFilename, _claim_id, df_prediction_output, df_evidence)
            
    except Exception as ex:
        print("Exception occured but will continue: " + str(ex))
        pass

    return render_template('result.html', input_claim=_input_claim, extracted_keywords_input_claim=_extracted_keywords_input_claim, 
                        classifier_result=_classifier_result, generated_justification = _generated_justification, extracted_keywords_generated_justification = _extracted_keywords_generated_justification, similarity_claims = "")

@app.route('/feedback', methods=['POST'])
def get_feedback():
    
    try:
        #Three feedbacks to be taken
        # 1. Veracity Feedback
        _veracity_feedback_option = ""
        _veracity_feedback_option = request.form['veracityfeedback']
        print("Veracity Feedback: " + str(_veracity_feedback_option))
        if (_veracity_feedback_option == "true"):
            _feedback_veracity = "0" #label 0: true claim
        if (_veracity_feedback_option == "fake"):
            _feedback_veracity = "1" #label 1: fake claim
        if (_veracity_feedback_option == "unsure"):
            _feedback_veracity = "Unsure" #no label
        if (_veracity_feedback_option == ""):
            _feedback_veracity = "No Feedback"
        # 2. Justification Feedback
        _justification_feedback_option = ""
        _justification_feedback_option = request.form['justificationfeedback']
        print("Justification Feedback: " + str(_justification_feedback_option))
        # 3. Justification Text
        _justification_feedback_text =  ""
        _justification_feedback_text = request.form['justificationFeedbackText']
        if(len(_justification_feedback_text)<0):
            _justification_feedback_text = "no written justification"
        print("Justification Text: " + str(_justification_feedback_text))
        
        _feedback_text = ""
        
        _feedback_text = "Veracity Feedback: " + str(_feedback_veracity) + ", Justification Feedback: " + str(_justification_feedback_option) + ", Justification Text: " + str(_justification_feedback_text)

        #open file and update feedback
        outputFilename = session.get('OUTPUTFILENAME')
        existing_df = pd.read_csv(outputFilename, sep='\t')
        existing_df.columns = ['ClaimID' , 'InputClaim', 'InputClaimKeywords', 'ScoreMetrics', 'Accuracy', 'GeneratedJustification', 'JustificationKeywords', 'EvidenceSources', 'Feedback']

        #Search for current claim ID & Append existing df with feedback
        current_claim_ID = session.get('CLAIMID')

        existing_df.loc[existing_df.ClaimID == current_claim_ID,'Feedback'] = _feedback_text

        #Write the updated df to file
        existing_df.to_csv(outputFilename, sep='\t', index=False)

        print("File Updated with feedback")

        current_claim_id = session.get('CLAIMID')
        print("We still have the claim ID:" + str(current_claim_id))
        #after writing set the session claim id to empty
        
    except Exception as ex:
        print("Exception occured but will continue: " + str(ex))
        pass


    print("Inside feedback function")
    return render_template('feedback.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port='5001')
