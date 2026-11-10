from smartticket.model import detect_sentiment, predict_ticket


def test_sentiment_positive():
    assert detect_sentiment("Thanks, this is excellent and helpful") == "positive"


def test_sentiment_negative():
    assert detect_sentiment("This is terrible and broken") == "negative"


def test_prediction_contract():
    result = predict_ticket("I was charged twice for my monthly subscription")
    assert result["category"] in {"billing", "technical", "account", "delivery", "cancellation"}
    assert result["priority"] in {"low", "medium", "high", "critical"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["needs_human_review"], bool)
