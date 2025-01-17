import faulthandler
from pathlib import Path
import time

import click

from src import Trainer, Tester, plotlosses


class Parameters:
    # Model Parameters
    input_size = 3
    # hidden_size = 200
    hidden_size = 64//2
    num_window_components = 10
    num_mixture_components = 20
    probability_bias = 1.0
    model_dir = Path("./trained_models/")
    idx = "n"
    # Dataset Parameters
    # dataset_dir = Path("../../Handwriting-Prediction-and-Synthesis/data/")
    dataset_dir = Path("../../data/")
    min_num_points = 300
    num_workers = 2
    # Training Parameters
    # num_epochs = 1000
    num_epochs = 1000
    # batch_size = 256
    batch_size = 128//4
    max_norm = 400
    # Optimizer Parameters
    optimizer = "Adam"
    learning_rate = 1.0e-4
    momentum = 0.9
    nesterov = True


@click.command()
@click.option("--train/--no-train", is_flag=True, default=True)
@click.option("--gpu/--no-gpu", is_flag=True, default=True)
@click.option("--idx", default=None)
def main(train: bool, gpu: bool, idx: str):
    print("Starting time: {}".format(time.asctime()))

    # To have a more verbose output in case of an exception
    faulthandler.enable()

    Parameters.train_model = train
    Parameters.use_gpu = gpu
    if idx is not None:
        Parameters.idx = idx

    if Parameters.train_model is True:
        # Instantiating the trainer
        trainer = Trainer(Parameters)
        # Training the model
        avg_losses = trainer.train_model()
        # Plot losses
        plotlosses(
            avg_losses,
            title="Average Loss per Epoch",
            xlabel="Epoch",
            ylabel="Average Loss",
        )

    # Instantiating the tester
    tester = Tester(Parameters)
    # Testing the model
    tester.test_random_sample()
    # Testing the model accuracy
    # test_losses = tester.test_model()

    print("Finishing time: {}".format(time.asctime()))


if __name__ == "__main__":
    main()
