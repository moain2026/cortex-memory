"""CORTEX v0.2 — no format tricks. Works on free natural language.
Supersession is generic: a new fact replaces an old one when they share an
entity AND their non-value wording is near-identical (same slot, new value)."""
import re, numpy as np, zlib
from collections import defaultdict
DIM=4096
STOP=set("the a an is are was of to and their his her now has have its new update changed moved".split())
def toks(t): return re.findall(r"\w+", t.lower())
def h(g): return zlib.crc32(g.encode())%DIM
def embed(t):
    v=np.zeros(DIM,np.float32)
    for w in toks(t):
        for g in {w}|{w[i:i+4] for i in range(max(1,len(w)-3))}: v[h(g)]+=1
    n=np.linalg.norm(v); return v/n if n else v
CAPSTOP={"the","you","new","revised","which","how","what","who","where","when"}
def ents(t): return {w.lower() for w in re.findall(r"\b[A-Z][a-z]+\d*\w*\b",t) if w.lower() not in CAPSTOP}
def is_value(w): return bool(re.search(r"\d",w)) and not re.match(r"[a-z]+\d{1,3}$",w)  # V12345 / numbers
def slot(t,E): return frozenset(w for w in toks(t) if w not in STOP and w not in E and not is_value(w))

class Cortex:
    def __init__(s, gate=True, supersede=True, spread=True, hebb=True, gate_thr=0.95, k=6, sthr=0.35):
        s.f=dict(gate=gate,supersede=supersede,spread=spread,hebb=hebb); s.thr=gate_thr; s.k=k; s.sthr=sthr
        s.cells={}; s.vecs={}; s.E={}; s.S={}; s.W=defaultdict(float); s.idx=defaultdict(set); s.n=0; s.rej=0
    def observe(s,t):
        v=embed(t); E=ents(t); S=slot(t,E)
        if s.f["gate"] and s.cells:
            ids=list(s.cells); sim=np.stack([s.vecs[i] for i in ids])@v
            if sim.max()>=s.thr: s.rej+=1; return
        if s.f["supersede"] and E and S:
            for e in E:
                for o in list(s.idx[e]):
                    if o in s.cells and s.E[o]==E and float(embed(' '.join(S))@embed(' '.join(s.S[o])))>=s.sthr: s._drop(o)
        i=s.n; s.n+=1; s.cells[i]=t; s.vecs[i]=v; s.E[i]=E; s.S[i]=S
        for e in E:
            for o in s.idx[e]: s.W[(i,o)]+=.5; s.W[(o,i)]+=.5
            s.idx[e].add(i)
    def _drop(s,o):
        for e in s.E[o]: s.idx[e].discard(o)
        for d in (s.cells,s.vecs,s.E,s.S): d.pop(o,None)
    def recall(s,q):
        ids=list(s.cells); sim=np.stack([s.vecs[i] for i in ids])@embed(q)
        act=defaultdict(float)
        for j in np.argsort(-sim)[:s.k]: act[ids[j]]+=float(sim[j])
        if s.f["spread"]:
            for e in ents(q):
                for c in s.idx.get(e,()): act[c]+=1.0
            qE=ents(q); seeds=[c for e in qE for c in s.idx.get(e,())] or sorted(act,key=act.get,reverse=True)[:3]
            for c in seeds:
                for e in s.E.get(c,()):
                    if e not in qE:
                        for c2 in s.idx.get(e,()):
                            if c2!=c: act[c2]+=0.9*float(s.vecs[c2]@embed(q))
            fr=dict(act); adj=defaultdict(list)
            for (a,b),w in s.W.items():
                if a in s.cells and b in s.cells: adj[a].append((b,w))
            for _ in range(2):
                nx=defaultdict(float)
                for a,x in fr.items():
                    for b,w in adj[a]: nx[b]+=.15*x*min(w,1)
                for b,x in nx.items(): act[b]+=x
                fr=nx
        top=sorted(act,key=act.get,reverse=True)[:s.k]
        if s.f["hebb"]:
            for a in top:
                for b in top:
                    if a!=b: s.W[(a,b)]+=.05
        return [s.cells[i] for i in top]
