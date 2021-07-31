from pathlib import Path
import json
import sys
import numpy as np
import os
from collections import Counter
import time
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

arg1 = sys.argv[1] #user provides the evaluation data
arg2 = sys.argv[2] #user provides the embeddings to be evaluated.
arg3 = Path("/project2/jevans/sloanlang/corpora/Expanded/large/reposVal.jsonl")

eval_path = Path(arg1)
embs_path = Path(arg2)

def load_jsonl(p):
    embeddings = {}
    with open(p, 'r') as fp:
        for line in tqdm(fp):
            data = json.loads(line)
            embeddings[data['name']] = data['embedding']
    return embeddings

def load_json(p):
    embeddings = json.load(open(p, 'r'))
    return embeddings

if ".jsonl" in str(embs_path):
    embeddings = load_jsonl(embs_path)
elif ".json" in str(embs_path):
    embeddings = load_json(embs_path)
else:
    print("Incorrect embeddings format")
print("Loaded Trained Embeddings.")
    
dataXY = json.load(open(eval_path, 'r'))
print("Loaded evaluation data.")

emb_dim = None
for r in embeddings:
    emb_dim = len(embeddings[r])
    break

random_embeddings = {}
for r in embeddings:
    random_embeddings[r] = np.random.rand(emb_dim)
    
X = dataXY['X']
Y = dataXY['Y']

for x in X:
    if x not in embeddings:
        embeddings[x] = np.zeros(emd_dim).tolist()
    if x not in random_embeddings:
        random_embeddings[x] = np.random.rand(emb_dim)

def evaluate(pred, true):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = 0
    pos = 0
    for a,b in zip(pred.flatten(), true.flatten()):
        if b == 1:
            pos += 1
        if a == 1 and b == 1:
            tp += 1
        if a == 0 and b == 0:
            tn += 1
        if a == 0 and b == 1:
            fn += 1
        if a == 1 and b == 0:
            fp += 1
        total += 1
    prec = 0 if tp == 0 else tp / (tp+fp)
    rec = 0 if tp == 0 else tp / (tp + fn)
    return pos, tp/total, tn/total, fp/total, fn/total, prec, rec

def evalRange(X, Y, embeddings, low, high):
    counts = Counter(Y)
    topN = sorted(counts.items(), key=lambda x: x[1], reverse=True)[low:high]
    topN = set([x for x,y in topN])
    topX = []
    topY = []
    for x,y in zip(X,Y):
        if y in topN:
            topX.append(x)
            topY.append(y)
            
    r2l = {}
    for x,y in zip(topX, topY):
        if x not in r2l:
            r2l[x] = []
        r2l[x].append(y)
        
    multiX = []
    multiY = []
    for r in r2l:
        multiX.append(r)
        multiY.append(r2l[r])
        
    labeler = MultiLabelBinarizer()
    labels = labeler.fit_transform(multiY)
    embX = [embeddings[x] if x in embeddings else np.random.randn(emb_dim) for x in multiX]
    
    X, Y = np.array(embX), np.array(labels)
    
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X, Y)
    pred = neigh.predict(X)
    
    pos, tpr, tnr, fpr, fnr, prec, rec = evaluate(pred, Y)
    if prec == 0 and rec == 0:
        return 0, 0, 0
    else:
        return ((2 * prec * rec)/(prec + rec)), prec, rec
    
print("#############")
print("EVALUATING TRAINED EMBEDDINGS:")
print("###")
f1, prec, rec = evalRange(X,Y,embeddings,0,100)
print(f"p:{prec:.4f} r:{rec:.4f} f1:{f1:.4f}")
    
print()

print("#############")
print("EVALUATING RANDOM EMBEDDINGS:")
print("###")
f1, prec, rec = evalRange(X,Y,random_embeddings,0,100)
print(f"p:{prec:.4f} r:{rec:.4f} f1:{f1:.4f}")