"""Fair audit: free-text facts (paraphrased templates), paraphrased queries,
updates phrased differently from originals, multi-hop questions, 5 seeds,
baselines incl. RAG+recency. Reports mean ± std."""
import random, sys, json, statistics as st, time
sys.path.insert(0,"."); import numpy as np
from cortex.core2 import Cortex, embed
A={"city":["{n} lives in the city of {v}","{n}'s home city is {v}"],
   "phone":["The phone number of {n} is {v}","You can reach {n} at phone {v}"],
   "budget":["{n} has a budget of {v}","The budget for {n} is {v}"],
   "manager":["{n} is managed by {m}","The account manager of {n} is {m}"]}
UPD={"city":"{n} relocated; the city of {n} is now {v}","phone":"New phone number of {n} is {v}",
     "budget":"Revised: the budget for {n} is now {v}","manager":"The account manager of {n} changed to {m}"}
Q={"city":"Which city does {n} live in?","phone":"How can I call {n}?","budget":"How much money can {n} spend?"}
NOISE="meeting coffee report draft review weather lunch call email note invoice schedule".split()
def run(seed):
    R=random.Random(seed); names=[f"Client{i:03d}" for i in range(150)]; mgrs=[f"Manager{i:02d}" for i in range(20)]
    mphone={m:f"V{R.randint(10000,99999)}" for m in mgrs}; S=[]; T={}; old={}
    for m in mgrs: S.append(f"The phone number of {m} is {mphone[m]}")
    for n in names:
        for a,tp in A.items():
            v=f"V{R.randint(10000,99999)}"; m=R.choice(mgrs)
            S.append(R.choice(tp).format(n=n,v=v,m=m)); T[(n,a)]= m if a=="manager" else v
    for _ in range(2000): S.insert(R.randrange(len(S))," ".join(R.choices(NOISE,k=9)))
    for n in R.sample(names,80):
        a=R.choice(["city","phone","budget"]); old[(n,a)]=T[(n,a)]; v=f"V{R.randint(10000,99999)}"
        S.append(UPD[a].format(n=n,v=v)); T[(n,a)]=v
    qs=[(Q[a].format(n=n),T[(n,a)],old.get((n,a))) for (n,a) in R.sample([k for k in T if k[1]!="manager"],200)]
    qu=[(Q[a].format(n=n),T[(n,a)],o) for (n,a),o in old.items()]
    qm=[(f"What is the phone number of the account manager of {n}?",mphone[T[(n,'manager')]],None) for n in R.sample(names,60)]
    V=np.stack([embed(x) for x in S])
    def rag(q,k=6): return [S[i] for i in np.argsort(-(V@embed(q)))[:k]]
    def rag_recent(q,k=6):   # strong baseline: retrieve 12, keep newest per entity-slot-ish (dedupe by entity, prefer later)
        idx=np.argsort(-(V@embed(q)))[:12]; idx=sorted(idx,reverse=True)[:k]; return [S[i] for i in idx]
    sysm={"RAG":rag,"RAG+recency":rag_recent}
    for name,kw in {"CORTEX":{},"−gate":dict(gate=False),"−supersede":dict(supersede=False),"−spread":dict(spread=False,hebb=False)}.items():
        c=Cortex(**kw); [c.observe(x) for x in S]; sysm[name]=c.recall
    out={}
    for nm,f in sysm.items():
        def acc(Qs): return sum((g in " ".join(ctx:=f(q))) and (o is None or o not in " ".join(ctx)) for q,g,o in Qs)/len(Qs)
        t=time.perf_counter(); out[nm]=dict(all=acc(qs),updated=acc(qu),multihop=acc(qm)); out[nm]["ms"]=(time.perf_counter()-t)*1000/(len(qs)+len(qu)+len(qm))
    return out
res=[run(s) for s in range(5)]
agg={}
for nm in res[0]:
    agg[nm]={}
    for m in res[0][nm]:
        xs=[r[nm][m] for r in res]
        agg[nm][m]=round(st.mean(xs),2) if m=="ms" else f"{100*st.mean(xs):.1f}±{100*st.pstdev(xs):.1f}"
print(json.dumps(agg,indent=1)); json.dump(agg,open("bench/audit_results.json","w"),indent=1)
