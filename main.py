import argparse
from train import train_model
from predict import predict_job_posting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake Job Posting Detector")
    parser.add_argument("--mode", choices=["train", "predict"], required=True, help="Mode: train or predict")
    parser.add_argument("--input", type=str, help="Input for prediction")

    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "predict":
        if args.input:
            result = predict_job_posting(args.input)
            print(f"Prediction: {result}")
        else:
            print("Please provide an input for prediction.")