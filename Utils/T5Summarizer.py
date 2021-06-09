import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import sklearn as sk
#from rouge_score import rouge_scorer

from transformers import T5Tokenizer, T5ForConditionalGeneration

import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.cuda.current_device()

TRAIN_BATCH_SIZE = 2    # input batch size for training (default: 64)
TEST_BATCH_SIZE = 2    # input batch size for testing (default: 1000)
TEST_EPOCHS = 1        # number of epochs to train (default: 10)
VAL_EPOCHS =  4
LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
SEED = 42               # random seed (default: 42)
MAX_LEN = 512
SUMMARY_LEN = 150 

torch.manual_seed(SEED) # pytorch random seed
np.random.seed(SEED) # numpy random seed
torch.backends.cudnn.deterministic = True

tokenizer = T5Tokenizer.from_pretrained("t5-base")

#create test dataloader
test_params = {
    'batch_size': TEST_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 0
    }

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')
      
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }
    
def generate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    rscores = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            #if _%100==0:
                #print(f'Completed {_}')
            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def getsummaryusingT5(_df_predictionset):

    df = _df_predictionset[['Crawled Article Text', 'Crawled Article Text']]
    
    print(df.head())
    df.columns = ['text','ctext']
    df.ctext = 'summarize: ' + df.ctext
    test_dataset=df.reset_index(drop=True)
    print(len(test_dataset))

    #Create Test Set
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    test_loader = DataLoader(test_set, **test_params)

    device = torch.device('cuda')
    fine_tuned_T5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    fine_tuned_T5_model = fine_tuned_T5_model.to(device)
    torch.cuda.empty_cache()
    path = "TrainedModels/T5NewsSummary_ds_weights_30_lr-0.0001.pt"
    fine_tuned_T5_model.load_state_dict(torch.load(path))
    
    device = torch.device('cuda')
    TEST_EPOCHS = 1
    finetunedT5_liar_summaries_df = {}
    print('FineTuned T5 Model')
    for epoch in range(TEST_EPOCHS):
        print("Generating Summaries")
        generated_text, actual_text = generate(epoch, tokenizer, fine_tuned_T5_model, device, test_loader)
        finetunedT5_liar_summaries_df = pd.DataFrame({'Generated Text':generated_text,'Actual Text':actual_text})
        #final_df.to_csv('predictions.csv')
        print("Summaries generated")
    
    _summary = ""
    for i,text in enumerate(finetunedT5_liar_summaries_df['Generated Text']):
        generated = finetunedT5_liar_summaries_df['Generated Text'][i]
        _summary = _summary + "..." + generated
    
    return _summary 