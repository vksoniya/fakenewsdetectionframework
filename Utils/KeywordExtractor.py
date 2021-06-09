#Extract Keyword:
from keybert import KeyBERT
model_keywords = KeyBERT('distilbert-base-nli-mean-tokens')

def extract_keywords(InputClaimFromUser):
    #Extract only trigrams
    input_keywords = model_keywords.extract_keywords(InputClaimFromUser, keyphrase_ngram_range=(3, 3), stop_words='english')
    print("Extracted Keywords from Input Claim are: " + str(input_keywords))

    return input_keywords