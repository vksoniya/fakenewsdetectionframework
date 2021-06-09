import pandas as pd
import numpy as np
import csv
import sklearn as sk
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

BATCH_SIZE = 16

def create_test_dataloader(df_crawled_articles, InputClaimFromUser):
    column_names = ["Title", "Text", "label"]
    df_input = pd.DataFrame(np.nan, index=range(0,len(df_crawled_articles["Crawled Article Text"])), columns=column_names)

    for i in range(len(df_crawled_articles["Crawled Article Text"])):
        df_input['Title'][i] = InputClaimFromUser
        df_input['Text'][i] = df_crawled_articles["Crawled Article Text"][i]

    df_input['label'] = 0 #Assuming the input to be true

    df_input

    final_test_path = "Datasets/final_test.tsv"
    df_input.to_csv(final_test_path, sep='\t', index=False)

    ctest_path = "Datasets/final_test.tsv"
    test_FINAL_df = pd.read_csv(ctest_path, sep='\t')
    test_FINAL_df.columns = ["Statement", "Justification", "Label"]
    test_FINAL_df.dropna()
    print("Length of Test Set:" + str(len(test_FINAL_df)))
    
    if(len(test_FINAL_df) > 0):
    
        test_FINAL_df.head()

        final_text_test = list(test_FINAL_df['Statement'])
        final_justification_test = list(test_FINAL_df['Justification'])
        final_labels_test = list(test_FINAL_df['Label'])

        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        ctext_tokens_test = tokenizer.batch_encode_plus(
            final_text_test,
            max_length = 100,
            padding=True,
            truncation=True
        )

        cjustification_tokens_test = tokenizer.batch_encode_plus(
            final_justification_test,
            max_length = 400,
            padding=True,
            truncation=True
        )

        ctext_seq_test = torch.tensor(ctext_tokens_test['input_ids'])
        cjustification_seq_test = torch.tensor(cjustification_tokens_test['input_ids'])
        ctest_seq = torch.cat((ctext_seq_test, cjustification_seq_test), 1)

        ctext_mask_test = torch.tensor(ctext_tokens_test['attention_mask'])
        cjustification_mask_test = torch.tensor(cjustification_tokens_test['attention_mask'])
        ctest_mask = torch.cat((ctext_mask_test, cjustification_mask_test), 1)

        ctest_y = torch.tensor(final_labels_test)

        ctest_data = TensorDataset(ctest_seq, ctest_mask, ctest_y)
        ctest_sampler = SequentialSampler(ctest_data)
        ctest_dataloader = DataLoader(ctest_data, sampler = ctest_sampler, batch_size=BATCH_SIZE)
        
    else:
        ctest_dataloader = ""
        final_labels_test = ""

    return ctest_dataloader, final_labels_test

