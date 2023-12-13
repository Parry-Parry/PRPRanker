from typing import Dict
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier import TransformerBase
import torch
import pandas as pd
import numpy as np
from pyterrier.model import add_ranks

class BayesRank(TransformerBase):
    def __init__(self, model_store : str, encoder, tokenizer, embedding_offsets : Dict[str, int], mem_file):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_store).to(self.device)
        self.encoder = encoder.to(self.device)
        self.tokenizer = tokenizer
        self.embedding_offsets = embedding_offsets
        self.mem_file = mem_file
        stem_name = f"org.terrier.terms.PorterStemmer"
        self.stemmer = pt.autoclass(stem_name)().stem
        self.stopwords = pt.autoclass("org.terrier.terms.Stopwords")(None).isStopword
    
    def score_doc(self, query, doc):
        e_doc = self.encoder(self.tokenizer(doc).input_ids)[0]

        q_terms = [self.stemmer(term) for term in query.split() if not self.stopwords(term)]
        d_terms = [self.stemmer(term) for term in doc.split() if not self.stopwords(term)]

        d_terms = [term for term in d_terms if term in q_terms]
        
        score = 0
        for term in d_terms:
            offset = self.embedding_offsets[term]
            e_term = self.mem_file.get(offset, 768)
            p, q = self.model(e_term, e_doc)
            doc_score = np.log(p*(1-q) / q(1-p))
            score += doc_score
        return score
    
    def transform(self, topics_and_docs : pd.DataFrame):
        output = topics_and_docs.copy()
        output["score"] = output.apply(lambda row: self.score_doc(row["query"], row["docno"]), axis=1)
        return add_ranks(output)

         
            



