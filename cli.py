"""Small command-line interface."""
import argparse
import json
from smartticket.model import predict_ticket, train_models


def main():
    parser = argparse.ArgumentParser(description="SmartTicket AI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Train models")
    predict = sub.add_parser("predict", help="Classify a support ticket")
    predict.add_argument("message", help="Ticket text")
    args = parser.parse_args()

    if args.command == "train":
        print(json.dumps(train_models(), indent=2))
    else:
        print(json.dumps(predict_ticket(args.message), indent=2))


if __name__ == "__main__":
    main()
