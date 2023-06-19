from src.train import train
import argparse


def create_train_parser():
    my_parser = argparse.ArgumentParser(description='Script used for retraining a model with a specific Hyperparameter set and save the trained model')

    my_parser.add_argument('--batch_size',
                           type=int,
                           help='Number patches per batch', default=50)

    my_parser.add_argument('--encoder_name',
                           type=str,
                           help='Name of an encoder (feature extractor), implemented: resnet18, resnet34, resnet50, resnet101')

    my_parser.add_argument('--learning_rate',
                           type=float,
                           help='Learning rate', default=1e-2)

    my_parser.add_argument('--training_strategy',
                           type=str,
                           help="Training Strategy: sharp patches only, or combined (sharp + blurry)", default="sharp")

    args = my_parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_train_parser()
    train(args)
