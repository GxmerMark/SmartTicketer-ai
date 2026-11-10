from smartticket.model import train_models
import json

if __name__ == "__main__":
    print(json.dumps(train_models(), indent=2))
