from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import sys

from sklearn.metrics.pairwise import cosine_similarity

# LOAD EVERYTHING

arg1 = sys.argv[1] #user provides the evaluation data
arg2 = sys.argv[2] #user provides the embeddings to be evaluated.

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
    
    
repos = []
for task in dataXY:
    source = task['source']
    targets = task['positive'] + task['negative']
    repos.append(source)
    repos.extend(targets)
repos = set(repos)

for x in repos:
    if x not in embeddings:
        embeddings[x] = np.zeros(emd_dim).tolist()
    if x not in random_embeddings:
        random_embeddings[x] = np.random.rand(emb_dim)


# SCRIPTS

def eval_mrr(true_, pred_):
    ranks = []
    for i, x in enumerate(pred_):
        if x in true_:
            ranks.append(i+1)
            break
    return np.mean([1/x for x in ranks])

def eval_map(true_, pred_):
    precs = []
    for i, x in enumerate(pred_):
        if x in true_:
            precs.append((len(precs) + 1) / (i+1))
    return np.mean(precs)

def eval_p5(true_, pred_):
    hits = []
    for i, x in enumerate(pred_):
        if i >= 5:
            break
        if x in true_:
            hits.append(x)
    return len(hits) / 5

# RUN THE TESTS

tests = [eval_mrr, eval_map, eval_p5]
test_names = ['MRR', 'MAP', 'P@5']

# DO THE REGULAR EMBEDDINGS

def linear_sim(x,y):
    return -np.linalg.norm(x-y)

sim = cosine_similarity

print("#############")
print("EVALUATING TRAINED EMBEDDINGS, COSINE:")
print("###")

for test, tn in zip(tests, test_names):

    scores = []
    for task in tqdm(dataXY):
        source = task['source']
        targets = task['positive'] + task['negative']
        pos = set(task['positive'])

        src = np.array(embeddings[source]).reshape(1,-1)
        tgts = [(x, sim(np.array(embeddings[x]).reshape(1,-1), src)) for x in targets]
        tgts = [(x,y.item()) for x,y in tgts]
        ranking = sorted(tgts, key=lambda x: x[1], reverse=True)
        ranking = [x for x,y in ranking]

        true_ = set(task['positive'])
        pred_ = ranking

        scores.append(test(true_, pred_) / test(true_, true_))
    print(f"{tn}: {np.mean(scores)}")
        

print()
print("#############")
print("EVALUATING RANDOM EMBEDDINGS, COSINE:")
print("###")
    
for test, tn in zip(tests, test_names):

    scores = []
    for task in tqdm(dataXY):
        source = task['source']
        targets = task['positive'] + task['negative']
        pos = set(task['positive'])

        src = np.array(random_embeddings[source]).reshape(1,-1)
        tgts = [(x, sim(np.array(random_embeddings[x]).reshape(1,-1), src)) for x in targets]
        tgts = [(x,y.item()) for x,y in tgts]
        ranking = sorted(tgts, key=lambda x: x[1], reverse=True)
        ranking = [x for x,y in ranking]

        true_ = set(task['positive'])
        pred_ = ranking

        scores.append(test(true_, pred_) / test(true_, true_))
    print(f"{tn}: {np.mean(scores)}")
       