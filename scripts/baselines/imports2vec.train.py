from pathlib import Path
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from multiprocessing import cpu_count
import sys

import logging
logging.basicConfig(level=logging.INFO)

trainingData = Path(sys.argv[1])

doc_ids = []
imports = []
with open(trainingData, 'r') as fp:
    for line in tqdm(fp):
        data = json.loads(line)
        repo = data['name']
        for f in data['imports']['files']:
            imps = data['imports']['files'][f]
            doc_ids.append(repo)
            imports.append(imps)
            
longest = max(map(len, imports))
docnames = sorted(list(set(doc_ids)))
d2i = { d:i for i,d in enumerate(docnames) }
i2d = { i:d for i,d in enumerate(docnames) }
json.dump(i2d, open("i2d.json", 'w'))

documents = [TaggedDocument(doc, [d2i[d]]) for d, doc in zip(doc_ids, imports)]

model = Doc2Vec(documents, vector_size=100, window=longest, min_count=3, workers=4, iter=100)
model.save(sys.argv[2])