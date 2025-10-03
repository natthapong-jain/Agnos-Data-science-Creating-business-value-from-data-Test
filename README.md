# Agnos Symptom Recommender — v2 (demo-conditional)

This build **uses the uploaded file**: `[CONFIDENTIAL] AI symptom picker data (Agnos candidate assignment).xlsx`

## What's new in v2
- Adds **demo-conditional co-occurrence** `P(b|a, gender, age_bin)`
- Final score: `final = α*(β*P(b|a,demo) + (1-β)*P(b|a)) + (1-α)*P(sym|demo)`
  - Defaults: α=0.6, β=0.7 — can be tuned per request
- Metrics (LOO, n=443): Recall@3=0.885, Recall@5=0.950, MAP@5=0.782, NDCG@5=0.825

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn pydantic
uvicorn app:app --reload --port 8000
```

Keep `app.py`, `symptom_model_v2.json`, `eval_summary.csv` in the same directory.

## Endpoints
- `POST /recommend` — body: `{gender, age, selected_symptoms:[], top_k?, alpha?, beta?}`
- `POST /recommend/explain` — returns per-item: `co_global`, `co_demo`, `prior_demo`, `final`
- `GET /rules?symptom=ไอ` — global neighbors; add `&gender=male&age=28` to see demo-conditional
- `GET /vocab?q=ไอ`
- `GET /metrics`
- `GET /healthz`

## Example
```bash
# Compare male vs female with same context
curl -X POST http://localhost:8000/recommend -H 'Content-Type: application/json'   -d '{"gender":"male","age":28,"selected_symptoms":["ไอ"],"top_k":10}'

curl -X POST http://localhost:8000/recommend -H 'Content-Type: application/json'   -d '{"gender":"female","age":28,"selected_symptoms":["ไอ"],"top_k":10}'

# Increase demographic influence
curl -X POST http://localhost:8000/recommend -H 'Content-Type: application/json'   -d '{"gender":"female","age":28,"selected_symptoms":["ไอ"],"top_k":10,"alpha":0.4,"beta":0.9}'
```
# Agnos-Data-science-Creating-business-value-from-data-Test
