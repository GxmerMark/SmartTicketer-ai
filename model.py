"""Training and inference utilities for SmartTicket AI."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tickets.csv"
MODEL_DIR = ROOT / "artifacts"
CATEGORY_MODEL = MODEL_DIR / "category_model.joblib"
PRIORITY_MODEL = MODEL_DIR / "priority_model.joblib"


def _pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True,
                                  strip_accents="unicode", sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train_models(data_path: Path = DATA_PATH) -> dict:
    """Train category and priority classifiers and save them locally."""
    df = pd.read_csv(data_path).dropna(subset=["message", "category", "priority"])
    if len(df) < 12:
        raise ValueError("Training data must contain at least 12 valid rows.")

    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
        df["message"].astype(str),
        df["category"].astype(str),
        df["priority"].astype(str),
        test_size=0.25,
        random_state=42,
        stratify=df["category"].astype(str),
    )

    category_model = _pipeline()
    priority_model = _pipeline()
    category_model.fit(X_train, y_cat_train)
    priority_model.fit(X_train, y_pri_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(category_model, CATEGORY_MODEL)
    joblib.dump(priority_model, PRIORITY_MODEL)

    cat_pred = category_model.predict(X_test)
    pri_pred = priority_model.predict(X_test)
    return {
        "rows": int(len(df)),
        "category_accuracy": round(float(accuracy_score(y_cat_test, cat_pred)), 4),
        "priority_accuracy": round(float(accuracy_score(y_pri_test, pri_pred)), 4),
        "category_report": classification_report(y_cat_test, cat_pred, zero_division=0),
        "priority_report": classification_report(y_pri_test, pri_pred, zero_division=0),
        "test_rows": int(len(X_test)),
    }


def load_models():
    if not CATEGORY_MODEL.exists() or not PRIORITY_MODEL.exists():
        train_models()
    return joblib.load(CATEGORY_MODEL), joblib.load(PRIORITY_MODEL)


def predict_ticket(message: str) -> dict:
    if not message or not message.strip():
        raise ValueError("Message cannot be empty.")
    category_model, priority_model = load_models()
    cleaned = message.strip()
    category = str(category_model.predict([cleaned])[0])
    priority = str(priority_model.predict([cleaned])[0])
    cat_probs = category_model.predict_proba([cleaned])[0]
    confidence = float(max(cat_probs))
    sentiment = detect_sentiment(cleaned)
    department = {
        "billing": "Billing",
        "technical": "Technical Support",
        "account": "Account Services",
        "delivery": "Logistics",
        "cancellation": "Retention",
    }.get(category, "Customer Support")
    return {
        "category": category,
        "priority": priority,
        "sentiment": sentiment,
        "department": department,
        "confidence": round(confidence, 4),
        "needs_human_review": confidence < 0.55 or priority == "critical",
    }


def detect_sentiment(message: str) -> str:
    """Transparent rule-based sentiment baseline; replace with a trained model for production."""
    positive = {"thanks", "thank", "great", "excellent", "love", "helpful", "happy", "good", "resolved"}
    negative = {"angry", "terrible", "bad", "broken", "frustrated", "hate", "worst", "unacceptable", "failed"}
    words = set(message.lower().split())
    score = len(words & positive) - len(words & negative)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"
