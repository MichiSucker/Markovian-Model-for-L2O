from experiments.lasso.training import set_up_and_train_algorithm
from experiments.lasso.evaluation import evaluate_algorithm
from experiments.lasso.plotting import create_evaluation_plot
from pathlib import Path
import torch


def create_folder_for_experiment(path_to_experiment_folder: str) -> str:
    path_of_experiment = path_to_experiment_folder + "/lasso/"
    Path(path_of_experiment).mkdir(parents=True, exist_ok=True)
    return path_of_experiment


def run(path_to_experiment_folder: str) -> None:

    print("Starting lasso experiment.")

    # This is pretty important again. Also, it makes sure that all tensor types do match.
    torch.set_default_dtype(torch.float64)

    path_of_experiment = create_folder_for_experiment(path_to_experiment_folder)
    print("\tStarting training.")
    set_up_and_train_algorithm(path_of_experiment=path_of_experiment)
    print("\tFinished training.")
    print("\tStarting evaluation.")
    evaluate_algorithm(path_of_experiment=path_of_experiment, loading_path=path_of_experiment + 'data/')
    print("\tFinished evaluation.")
    print("\tCreating evaluation plot.")
    create_evaluation_plot(loading_path=path_of_experiment + 'data/', path_of_experiment=path_of_experiment)

    print("Finished lasso experiment.")