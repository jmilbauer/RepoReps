from pathlib import Path
import json
import csv
from tqdm import tqdm

import csrgraph as cg
import nodevectors

import logging
logging.basicConfig(level=logging.INFO)

arg1 = "../../corpus/reposAll.jsonl"
elpath = "tmp_edgelist.csv"

trainingData = Path(arg1)

r2c = {}
with open(trainingData, 'r') as fp:
    for line in tqdm(fp):
        data = json.loads(line)
        repo = data['name']
        contributors = data['contributors']
        contributors = [tuple(x) for x in contributors]
        r2c[repo] = contributors
        
c2r = {}
for r in tqdm(r2c):
    for c in r2c[r]:
        if c not in c2r:
            c2r[c] = set([])
        c2r[c].add(r)
        
r2r = {}
for r1 in tqdm(r2c):
    for c in r2c[r1]:
        for r2 in c2r[c]:
            if r1 == r2:
                continue
            key = (r1,r2)
            if key not in r2r:
                r2r[key] = 0
            else:
                r2r[key] += 1
                
goodkeys = set([])
badkeys = set([])

for a,b in r2r:
    if (b,a) in goodkeys:
        badkeys.add((a,b))
    else:
        goodkeys.add((a,b))
        
repos = r2c.keys()
r2i = { r:i for i,r in enumerate(repos)}
i2r = { i:r for i,r in enumerate(repos)}

json.dump(i2r, open("i2r.json", 'w'))

with open(elpath, 'w') as fp:
    for a,b in tqdm(goodkeys):
        out = f"{a},{b},{r2r[(a,b)]}"
        fp.write(f"{out}\n")
        
python_network = cg.read_edgelist(elpath, directed=False, sep=',')

node2vec_python = nodevectors.Node2Vec(n_components=100,walklen=80,epochs=20) 
node2vec_python_embeddings = node2vec_python.fit(python_network)
node2vec_python.save('../../models/net2vec.nodevectors.model')