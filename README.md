# SmartTicket AI 🎫

**An explainable NLP-based customer support ticket triage system.**

SmartTicket AI uses TF-IDF and Logistic Regression to classify incoming support messages and estimate their priority. It also provides a lightweight sentiment baseline, department routing, confidence scores, and a FastAPI interface.

## Features

- Ticket category prediction: `billing`, `technical`, `account`, `delivery`, `cancellation`
- Priority prediction: `low`, `medium`, `high`, `critical`
- Sentiment signal: positive / neutral / negative (transparent keyword baseline)
- Prediction confidence and human-review flag
- Trainable models and evaluation reports
- REST API with interactive Swagger docs
- CLI inference and automated tests
- No paid API key required

## Architecture

```text
Ticket message
     |
     v
Text cleaning + TF-IDF (unigrams and bigrams)
     |
     +----> Logistic Regression ----> Category
     |
     +----> Logistic Regression ----> Priority
     |
     +----> Keyword baseline --------> Sentiment
     |
     v
Department routing + confidence + review flag
     |
     v
JSON response / FastAPI
```

## Quick start

Python 3.10+ is recommended.

```bash
git clone https://github.com/YOUR_USERNAME/smartticket-ai.git
cd smartticket-ai

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py
python -m smartticket.cli predict "I was charged twice for my subscription"
```

On Android Termux, use `pkg install python git` first, then follow the same steps. If a dependency wheel is unavailable for your device, run it in a Python 3.10+ Linux environment.

## Run the API

```bash
uvicorn smartticket.api:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"message":"My payment was deducted twice and I need a refund"}'
```

Example response shape:

```json
{
  "category": "billing",
  "priority": "high",
  "sentiment": "negative",
  "department": "Billing",
  "confidence": 0.82,
  "needs_human_review": false
}
```

Predictions and confidence values depend on the training data and environment; the example is illustrative.

## Train and evaluate

```bash
python train.py
```

The training script performs a stratified train/test split, trains two pipelines, saves model artifacts under `artifacts/`, and prints accuracy and classification reports.

## Tests

```bash
python -m pytest -q
```

## Dataset

`data/tickets.csv` is a small, hand-authored illustrative dataset for demonstrating the workflow. It is not representative of real customer support traffic. Before real deployment, replace it with a larger, diverse, consented and appropriately anonymized dataset, then evaluate per-class precision, recall, F1, calibration, and performance across customer groups.

## Important limitations

- This is a portfolio/demo project, not a production customer-support decision system.
- The training dataset is intentionally small; metrics are not evidence of production accuracy.
- Priority labels are learned from examples and may be wrong. Critical cases should always be reviewed by a person.
- Sentiment is a simple keyword baseline, not a deep-learning sentiment model.
- Confidence is the model's probability estimate, not a guarantee of correctness.
- Add authentication, rate limiting, monitoring, privacy controls, and robust data validation before deployment.

## Roadmap

- [ ] Add a larger public benchmark dataset
- [ ] Compare Logistic Regression, Linear SVM, and transformer embeddings
- [ ] Add calibration and per-class error analysis
- [ ] Add multilingual ticket support
- [ ] Add a review dashboard and feedback-based retraining

## License

MIT
