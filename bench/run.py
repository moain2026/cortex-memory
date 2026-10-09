"""Benchmark: memory with fact UPDATES + distractor noise.
Answer is correct if the CURRENT value is in retrieved context and the stale one is not."""
import random, time, sys, json
sys.path.insert(0, ".")
from cortex.core import Cortex, embed
import numpy as np
random.seed(7)
N_ENT, N_NOISE = 120, 1500
attrs = ["city","phone","budget","deadline","favorite_style"]
vals = lambda: f"V{random.randint(10000,99999)}"
stream, truth, stale = [], {}, {}
names = [f"Client{i:03d}" for i in range(N_ENT)]
for n in names:
    for a in attrs:
        v = vals(); stream.append(f"{n} {a}:: {n} {a} is {v}"); truth[(n,a)] = v
noise_words = "meeting coffee report draft review weather lunch call email update note".split()
for _ in range(N_NOISE):
    stream.insert(random.randrange(len(stream)), " ".join(random.choices(noise_words, k=8)))
for n in random.sample(names, 60):                     # 60 updates (contradictions)
    a = random.choice(attrs); stale[(n,a)] = truth[(n,a)]
    v = vals(); stream.append(f"{n} {a}:: {n} {a} is {v}"); truth[(n,a)] = v
qs = [(n,a) for (n,a) in random.sample(list(truth), 200)] + list(stale)
tok = lambda xs: sum(len(x.split()) for x in xs)

def score(get):
    ok=okU=nU=0; T=0; toks=0
    for (n,a) in qs:
        q = f"What is {n} {a}?"; t0=time.perf_counter(); ctx = get(q); T += time.perf_counter()-t0
        blob = " ".join(ctx); toks += tok(ctx)
        good = truth[(n,a)] in blob and (stale.get((n,a)) not in blob if (n,a) in stale else True)
        ok += good
        if (n,a) in stale: nU += 1; okU += good
    return dict(accuracy=round(ok/len(qs),3), update_accuracy=round(okU/nU,3),
                avg_tokens=round(toks/len(qs)), ms_per_query=round(1000*T/len(qs),2))

res = {}
res["full_context"] = score(lambda q: stream)            # dump everything
vecs = np.stack([embed(x) for x in stream])
def rag(q, k=6):
    s = vecs @ embed(q); return [stream[i] for i in np.argsort(-s)[:k]]
res["plain_RAG_k6"] = score(rag)
cx = Cortex(); t0=time.perf_counter()
for x in stream: cx.observe(x)
cx.sleep(); ingest = time.perf_counter()-t0
res["CORTEX"] = score(cx.context)
res["CORTEX"]["cells_stored"] = len(cx.cells); res["CORTEX"]["rejected_by_gate"] = cx.rejected
res["meta"] = dict(stream_items=len(stream), queries=len(qs), update_queries=len(stale), ingest_s=round(ingest,2))
print(json.dumps(res, indent=2))
json.dump(res, open("bench/results.json","w"), indent=2)
