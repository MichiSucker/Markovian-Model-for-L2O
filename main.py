import unittest
import coverage
from experiments.quadratics.run_experiment import run as run_quadratics
from experiments.neural_network_full_batch.run_experiment import run as run_nn_full_batch
from experiments.image_processing.run_experiment import run as run_image_processing
from experiments.lasso.run_experiment import run as run_lasso
from experiments.neural_network_stochastic.run_experiment import run as run_stochastic_nn


def run_tests():

    cov = coverage.Coverage(omit=['*test_*', 'main.py', '__init__.py'])
    cov.start()

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir='tests/PacBayesOptimizationAlgorithm')
    runner = unittest.runner.TextTestRunner().run(test_suite)

    cov.stop()
    cov.save()
    cov.html_report()


def run_experiments():
    run_quadratics(path_to_experiment_folder='/home/michael/Desktop/JMLR_Markovian_Model/new_implementation')
    run_image_processing(path_to_experiment_folder='/home/michael/Desktop/JMLR_Markovian_Model/new_implementation')
    run_lasso(path_to_experiment_folder='/home/michael/Desktop/JMLR_Markovian_Model/new_implementation')
    run_nn_full_batch(path_to_experiment_folder='/home/michael/Desktop/JMLR_Markovian_Model/new_implementation')
    run_stochastic_nn(path_to_experiment_folder='/home/michael/Desktop/JMLR_Markovian_Model/new_implementation')


if __name__ == '__main__':

    run_experiments()
