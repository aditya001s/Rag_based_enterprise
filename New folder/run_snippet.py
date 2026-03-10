import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),'training'))
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from utils import Vocab
from train import RNNClassifier, LSTMClassifier, SelfAttentionClassifier

emb = np.load('training/embeddings.npy')
metas = pd.read_json('training/metas.json')
chunks = pd.read_csv('training/extracted_chunks.csv')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
emb_norm = emb / (norm(emb, axis=1, keepdims=True) + 1e-12)

queries = [
    "What are access control requirements?",
    "How should organizations handle incident response?",
    "What privacy controls support data minimization?"
]

models_to_check = ['rnn','lstm','attn']
ckpts = {}
for m in models_to_check:
    p = os.path.join('training','models', f'best_{m}.pt')
    if os.path.exists(p):
        ckpts[m]=p

print('Found checkpoints:', ckpts)
inst_models = {}
for name,p in ckpts.items():
    ck = torch.load(p, map_location='cpu')
    stoi = ck.get('vocab')
    vocab = Vocab()
    if isinstance(stoi, dict):
        vocab.stoi = dict(stoi)
        size = max(vocab.stoi.values())+1
        itos = [None]*size
        for tok, idx in vocab.stoi.items():
            if idx < size:
                itos[idx]=tok
        vocab.itos = [t if t is not None else vocab.unk_token for t in itos]
    else:
        vocab.build([''])
    st = ck['model_state']
    fc_key = None
    for k in st.keys():
        if k.endswith('fc.weight'):
            fc_key=k
            break
    if fc_key is None:
        num_classes = 2
    else:
        num_classes = st[fc_key].shape[0]
    vocab_size = len(vocab)
    if name=='rnn':
        inst = RNNClassifier(vocab_size, 128, 128, num_classes)
    elif name=='lstm':
        inst = LSTMClassifier(vocab_size, 128, 128, num_classes)
    else:
        inst = SelfAttentionClassifier(vocab_size, 128, 4, 128, num_classes)
    inst.load_state_dict(ck['model_state'])
    inst.eval()
    inst_models[name]= (inst, vocab)

for q in queries:
    print('\n=== QUERY:', q, '===')
    q_emb = model.encode([q], convert_to_numpy=True)
    qn = q_emb / (norm(q_emb, axis=1, keepdims=True) + 1e-12)
    sims = emb_norm @ qn[0]
    topk = sims.argsort()[-3:][::-1]
    print('\nTop passages:')
    for rank, idx in enumerate(topk, 1):
        meta = metas.iloc[idx]
        text = chunks.iloc[idx]['text'] if 'text' in chunks.columns else ''
        snippet = text.replace('\n',' ')[:800]
        print(f"[{rank}] score={float(sims[idx]):.4f} source={meta.get('source')} chunk={meta.get('chunk_id')}\n{snippet}\n")
    print('\nModel predictions:')
    for name,(mdl,vocab) in inst_models.items():
        ids = vocab.encode(q)
        if len(ids)==0:
            ids=[vocab.stoi.get(vocab.unk_token,1)]
        x = torch.tensor([ids], dtype=torch.long)
        lengths = torch.tensor([len(ids)], dtype=torch.long)
        with torch.no_grad():
            logits = mdl(x, lengths)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy().tolist()
            pred = int(torch.argmax(logits,dim=1).item())
        print(f"{name}: pred={pred} probs={probs}")
print('\nDone')
