from pathlib import Path
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from multiprocessing import cpu_count
import sys

from nltk.tokenize import sent_tokenize, word_tokenize

import logging
logging.basicConfig(level=logging.INFO)

trainingData = Path(sys.argv[1])

doc_ids = []
readmes = []
with open(trainingData, 'r') as fp:
    for line in tqdm(fp):
        data = json.loads(line)
        repo = data['name']
        doc_ids.append(repo)
        readme = data['readme'].lower()[:10000]
        tokens = [w for s in sent_tokenize(readme) for w in word_tokenize(s)]
        readmes.append(tokens)
            
docnames = sorted(list(set(doc_ids)))
d2i = { d:i for i,d in enumerate(docnames) }
i2d = { i:d for i,d in enumerate(docnames) }

documents = [TaggedDocument(doc, [d2i[d]]) for d, doc in zip(doc_ids, readmes)]
model = Doc2Vec(documents, vector_size=100, window=5, min_count=3, workers=4, iter=100)
model.save(sys.argv[2])