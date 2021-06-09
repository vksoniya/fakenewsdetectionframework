from transformers import pipeline, set_seed
from Utils.KeywordExtractor import extract_keywords
from Utils.WebCrawler import crawl_web
import spacy
import numpy as np
import pandas as pd

import en_core_web_lg
nlp = en_core_web_lg.load()
#nlp = spacy.load("en_core_web_lg")

def _load_gpt_model():
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    return generator


def generateExplantion(InputClaimFromUser):
    generator_gpt = _load_gpt_model()

    generated_justification = generator_gpt(InputClaimFromUser, max_length=350, num_return_sequences=1)
    gJustification = generated_justification[0]['generated_text']

    return gJustification

def extractEvidence(InputClaimFromUser, _keywords_justification):
    
    df_crawled_evidence = crawl_web(_keywords_justification)
    print("Web crawling for evidence completed:" +  str(len(df_crawled_evidence)))

    #Determine Similarity
    loadedClaimsText = list(df_crawled_evidence['Crawled Article Text'])
    loadedClaimsLink = list(df_crawled_evidence['Crawled Article Link'])
    loadedClaimsTitle = list(df_crawled_evidence['Crawled Article Title'])
    similarities = []

    print(len(loadedClaimsText))

    for i,claim in enumerate(loadedClaimsText):
        if(len(claim) > 0):
            score = nlp(InputClaimFromUser).similarity(nlp(claim))
            similarities.append(score)

    idx = np.argsort(similarities)

    df_similarclaims = pd.DataFrame(columns=['Similarity Score', 'Similar Claim Text', 'Similar Claim Title', 'Source'], index=range(len(idx)))

    c = 0
    for i in idx:
        df_similarclaims['Similarity Score'][c]= str(similarities[i])
        df_similarclaims['Similar Claim Text'][c] = str(loadedClaimsText[i])
        df_similarclaims['Similar Claim Title'][c] = str(loadedClaimsTitle[i])
        df_similarclaims['Source'][c] = str(loadedClaimsLink[i])
        c = c + 1

    df_evidence = df_similarclaims.sort_values('Similarity Score', ascending=False).reset_index(drop=True)
    print("Similarities computed:" +  str(len(df_crawled_evidence)))

    return df_evidence