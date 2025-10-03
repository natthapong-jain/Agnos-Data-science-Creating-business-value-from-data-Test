import json, math
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI(title="Agnos Symptom Recommender", version="2.0")

# ---- load model ----
with open("symptom_model_v2.json", "r", encoding="utf-8") as f:
    MODEL = json.load(f)

VOCAB = set(MODEL["vocab"])
SYM_COUNTS: Dict[str, int] = MODEL["sym_counts"]
COND_G: Dict[str, Dict[str, float]] = MODEL["cond_prob_global"]
COND_D: Dict[str, Dict[str, Dict[str, float]]] = MODEL["cond_prob_demo"]
PRIOR_D: Dict[str, Dict[str, float]] = MODEL["demo_prior"]
ALPHA_DEFAULT = MODEL["notes"]["alpha_default"]
BETA_DEFAULT = MODEL["notes"]["beta_default"]

def age_bin(age:int)->str:
    if age <= 12: return "0-12"
    if age <= 19: return "13-19"
    if age <= 34: return "20-34"
    if age <= 49: return "35-49"
    if age <= 64: return "50-64"
    return "65+"

def key_of(gender:str, age:int)->str:
    return f"{gender.lower()}|{age_bin(age)}"

def recommend(selected:List[str], gender:str, age:int, k:int=10, alpha:float=ALPHA_DEFAULT, beta:float=BETA_DEFAULT)->List[str]:
    sel = [s for s in selected if s in VOCAB]
    key = key_of(gender, age)
    # candidate generation
    cand = set()
    for s in sel:
        cand |= set(COND_G.get(s, {}).keys())
        cand |= set(COND_D.get(key, {}).get(s, {}).keys())
    if not cand:
        cand = set(VOCAB)
    scores: Dict[str, float] = {}
    for c in cand:
        if c in sel: 
            continue
        # mean over selected
        co_gl = sum(COND_G.get(s, {}).get(c, 0.0) for s in sel) / max(1, len(sel))
        co_dm = sum(COND_D.get(key, {}).get(s, {}).get(c, 0.0) for s in sel) / max(1, len(sel))
        prior = PRIOR_D.get(key, {}).get(c, 0.0)
        final = alpha*(beta*co_dm + (1-beta)*co_gl) + (1-alpha)*prior
        scores[c] = final
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [s for s,_ in ranked]

class RecommendRequest(BaseModel):
    gender: str
    age: int
    selected_symptoms: List[str]
    top_k: int = 10
    alpha: float = ALPHA_DEFAULT
    beta: float = BETA_DEFAULT

class RecommendResponse(BaseModel):
    recommendations: List[str]

@app.post("/recommend", response_model=RecommendResponse)
def recommend_api(req: RecommendRequest):
    recs = recommend(req.selected_symptoms, req.gender, req.age, k=req.top_k, alpha=req.alpha, beta=req.beta)
    return {"recommendations": recs}

@app.post("/recommend/explain")
def recommend_explain(req: RecommendRequest):
    sel = [s for s in req.selected_symptoms if s in VOCAB]
    key = key_of(req.gender, req.age)
    cand = set()
    for s in sel:
        cand |= set(COND_G.get(s, {}).keys())
        cand |= set(COND_D.get(key, {}).get(s, {}).keys())
    if not cand:
        cand = set(VOCAB)
    rows = []
    for c in cand:
        if c in sel: 
            continue
        co_gl = sum(COND_G.get(s, {}).get(c, 0.0) for s in sel) / max(1, len(sel))
        co_dm = sum(COND_D.get(key, {}).get(s, {}).get(c, 0.0) for s in sel) / max(1, len(sel))
        prior = PRIOR_D.get(key, {}).get(c, 0.0)
        final = req.alpha*(req.beta*co_dm + (1-req.beta)*co_gl) + (1-req.alpha)*prior
        rows.append({"symptom": c, "co_global": co_gl, "co_demo": co_dm, "prior_demo": prior, "final": final})
    rows.sort(key=lambda x: x["final"], reverse=True)
    return {"items": rows[: req.top_k]}

@app.get("/rules")
def rules(symptom: str = Query(...), gender: Optional[str] = None, age: Optional[int] = None):
    # if gender & age provided -> show demo-conditional; else global
    if gender is not None and age is not None:
        key = key_of(gender, age)
        neigh = COND_D.get(key, {}).get(symptom, {})
    else:
        neigh = COND_G.get(symptom, {})
    ranked = sorted(neigh.items(), key=lambda x: x[1], reverse=True)
    return {"symptom": symptom, "neighbors": [{"symptom": s, "weight": w} for s, w in ranked]}

@app.get("/vocab")
def vocab(q: Optional[str] = Query(default=None)):
    items = sorted(list(VOCAB))
    if q:
        items = [s for s in items if q in s]
    return {"count": len(items), "items": items}

@app.get("/metrics")
def metrics():
    import os, csv
    path = "eval_summary.csv"
    if not os.path.exists(path):
        return {"info": "eval_summary.csv not found next to app.py", "metrics": {}}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, {}) or {}
    # best-effort numeric cast
    def to_num(v):
        try:
            return float(v)
        except Exception:
            return v
    return {"metrics": {k: to_num(v) for k, v in row.items()}}

@app.get("/healthz")
def healthz():
    return {"ok": True}
