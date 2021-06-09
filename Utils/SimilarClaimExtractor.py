#This module will extract text using methods:
# -. Cosine Similarity using Spacy Library

import spacy
import numpy as np
import pandas as pd

import en_core_web_lg
nlp = en_core_web_lg.load()

# get similar text using cosine similarity (spacy)
# returns df containing similarity score and the text
def extractCosineSimilarityText(InputClaimFromUser, _df_crawled_articles):
    
    _text = list(_df_crawled_articles['Crawled Article Text'])
    _title = list(_df_crawled_articles['Crawled Article Title'])
    _link = list(_df_crawled_articles['Crawled Article Link'])
    similarities = []

    print(len(_text))

    for i,claim in enumerate(_text):
        if(len(claim) > 0):
            score = nlp(InputClaimFromUser).similarity(nlp(claim))
            similarities.append(score)

    idx = np.argsort(similarities)

   
    _df = pd.DataFrame(columns=['Similarity Score', 'Crawled Article Title', 'Crawled Article Text', 'Crawled Article Link'], index=range(len(idx)))

    c = 0
    for i in idx:
        _df['Similarity Score'][c]= str(similarities[i])
        _df['Crawled Article Text'][c] = str(_text[i])
        _df['Crawled Article Title'][c] = str(_title[i])
        _df['Crawled Article Link'][c] = str(_link[i])
        c = c + 1

    df_similartext = _df.sort_values('Similarity Score', ascending=False).reset_index(drop=True)
    print("Similarities computed:" +  str(len(df_similartext)))
    df_similartext = df_similartext.head(10)
    print("Similarities computed and only top 10 returned:" +  str(len(df_similartext)))

    return df_similartext