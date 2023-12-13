import torch
from transformers import BertTokenizer, BertModel
import pyterrier as pt
if not pt.started():
    pt.init()

def embed(vocabulary_file : str, model_id : str, out_dir : str):
    with open(vocabulary_file, 'r') as f:
        vocabulary = f.read().splitlines()
    model = BertModel.from_pretrained(model_id)
    tokenizer = BertTokenizer.from_pretrained(model_id)
    stemmer = pt.autoclass("PorterStemmer")().stem
    stopwords = pt.autoclass("org.terrier.terms.Stopwords")(None).isStopword

    for word in vocabulary:
        if stopwords(word) or stopwords(stemmer(word)):
            continue
        tokens = tokenizer.tokenize(stemmer(word))
        embedding = model(torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]))[0].detach().numpy()
    
