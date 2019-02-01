#Import and Conversion, Normalization of Data
from Open_conversion_data import load_csv
from Open_conversion_data import str_column_to_float
from Open_conversion_data import str_column_to_int
from Open_conversion_data import dataset_minmax
from Open_conversion_data import Normalize_Dataset

#Import seed for generating random data
from random import seed

#Algorithm evaluation with different steps
from Algorithm_test_harness import evaluate_algorithm_cv
from Algorithm_test_harness import evaluate_algorithm_tt_split
#Backpropagation algorithm
from Backpropagation_model import back_propagation_sgs

# Accuracy for assessment metrics
from Performance_assessment import getAccuracy


def main():
    # Test Backprop on Seeds dataset
    seed(1)
    # load and prepare data
    filename = 'wheat_seeds.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    Normalize_Dataset(dataset, minmax)

    # evaluate algorithm
    n_folds = 10
    l_rate = 0.5
    n_epoch = 200
    n_hidden = 10
    num_added_hidden_layers = 0
    #split = 0.6
    #n_splits = 2
    #scores = evaluate_algorithm_tt_split(dataset, back_propagation_sgs, split, n_splits,l_rate, n_epoch, n_hidden, num_added_hidden_layers)
    scores = evaluate_algorithm_cv(dataset, back_propagation_sgs, n_folds, getAccuracy,l_rate, n_epoch, n_hidden, num_added_hidden_layers)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

main()