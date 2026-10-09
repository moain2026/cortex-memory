"""CORTEX prototype: surprise gate + hippocampal graph (spreading activation,
Hebbian learning, decay) + sleep consolidation into cortical summaries.
Embeddings: char n-gram hashing (no external model) so it runs anywhere."""
import re, time, math
import numpy as np
from collections import defaultdict

DIM = 2048
def _tok(t): return re.findall(r"\w+", t.lower())
def embed(text):
    v = np.zeros(DIM, dtype=np.float32)
    for w in _tok(text):
        for g in {w} | {w[i:i+4] for i in range(max(1, len(w)-3))}:
            v[hash(g) % DIM] += 1.0
    n = np.linalg.norm(v); return v / n if n else v

def entities(text):
    """Keys a fact is about: capitalised words + 'key:' prefix."""
    m = re.match(r"\s*([\w\- ]+?)\s*::", text)
    keys = {m.group(1).strip().lower()} if m else set()
    return keys | {w.lower() for w in re.findall(r"\b[A-Z][\w\-]+", text)}

class Cell:
    __slots__ = ("id","text","vec","keys","t","uses")
    def __init__(s, i, text, t):
        s.id, s.text, s.vec, s.keys, s.t, s.uses = i, text, embed(text), entities(text), t, 0

class Cortex:
    def __init__(s, surprise=0.92, decay=0.97, k=6):
        s.cells, s.W = {}, defaultdict(float)      # W[(a,b)] synapse weight
        s.key_index = defaultdict(set)
        s.cortex = {}                               # key -> consolidated summary
        s.surprise, s.decay, s.k, s.clock, s.rejected = surprise, decay, k, 0, 0

    # ---- perception with surprise gate ----
    def observe(s, text):
        s.clock += 1
        v = embed(text)
        subj = text.split("::")[0].strip().lower() if "::" in text else None
        contradicts = subj and any(subj in s.key_index and s.key_index[subj] for _ in [0])
        if s.cells and not contradicts:   # a fact that contradicts memory is always surprising
            M = np.stack([c.vec for c in s.cells.values()])
            if float((M @ v).max()) >= s.surprise:     # already known -> not surprising
                s.rejected += 1; return None
        c = Cell(len(s.cells) + s.rejected, text, s.clock)
        # supersede: same subject key -> newer fact wins (temporal memory)
        for key in c.keys:
            for oid in list(s.key_index[key]):
                o = s.cells.get(oid)
                if o and o.text.split("::")[0] == c.text.split("::")[0] and "::" in c.text:
                    s._forget(oid)
            s.key_index[key].add(c.id)
        s.cells[c.id] = c
        for key in c.keys:                              # wire to same-entity cells
            for oid in s.key_index[key]:
                if oid != c.id: s.W[(c.id,oid)] += 0.5; s.W[(oid,c.id)] += 0.5
        return c.id

    def _forget(s, cid):
        c = s.cells.pop(cid, None)
        if not c: return
        for key in c.keys: s.key_index[key].discard(cid)
        for e in [e for e in s.W if cid in e]: del s.W[e]

    # ---- recall: seed by similarity, spread activation 2 hops ----
    def recall(s, query, k=None):
        k = k or s.k
        if not s.cells: return []
        ids = list(s.cells); M = np.stack([s.cells[i].vec for i in ids])
        sim = M @ embed(query)
        act = defaultdict(float)
        for j in np.argsort(-sim)[:3]: act[ids[j]] += float(sim[j])
        for kq in entities(query):
            for cid in s.key_index.get(kq, ()): act[cid] += 0.6
        adj = defaultdict(list)
        for (a,b),w in s.W.items(): adj[a].append((b,w))
        frontier = dict(act)
        for _ in range(2):
            nxt = defaultdict(float)
            for a,x in frontier.items():
                for b,w in adj[a]: nxt[b] += 0.5 * x * min(w,1.0)
            for b,x in nxt.items(): act[b] += x
            frontier = nxt
        top = sorted(act, key=act.get, reverse=True)[:k]
        for a in top:                                   # Hebbian: co-fired -> stronger
            s.cells[a].uses += 1
            for b in top:
                if a != b: s.W[(a,b)] += 0.1
        return [s.cells[i].text for i in top]

    # ---- sleep: decay unused synapses, consolidate hot entities ----
    def sleep(s):
        for e in list(s.W):
            s.W[e] *= s.decay
            if s.W[e] < 0.05: del s.W[e]
        for key, ids in s.key_index.items():
            if len(ids) >= 3:
                s.cortex[key] = " | ".join(s.cells[i].text for i in sorted(ids) if i in s.cells)

    def context(s, query):
        lines = s.recall(query)
        hot = [s.cortex[k] for k in entities(query) if k in s.cortex]
        return hot[:1] + lines
