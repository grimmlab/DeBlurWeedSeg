from src.train import retrain
import argparse


def create_train_parser():
    my_parser = argparse.ArgumentParser(description='Script used for retraining a model with a specific Hyperparameter set and save the trained model')

    my_parser.add_argument('--batch_size',
                           type=int,
                           help='Number patches per batch', default=50)

    my_parser.add_argument('--max_steps',
                           type=int,
                           help='Number of steps to train for', default=10)

    my_parser.add_argument('--encoder_name',
                           type=str,
                           help='Name of an encoder (feature extractor), implemented: resnet18, resnet34, resnet50, resnet101')

    my_parser.add_argument('--learning_rate',
                           type=float,
                           help='Learning rate', default=1e-2)

    args = my_parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_train_parser()
    retrain(args)
