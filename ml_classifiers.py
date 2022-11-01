# Author: Amish Mishra
# Date: October 7, 2022
# README: This file trains an ML classifier on the CGMH-training data. Instead of manually downsampling, I use
# SVC's in built function to deal with the imbalanced dataset

from sklearn.svm import SVC
import pandas
import pickle


# For reference: stages = ['rem', 'wake', 's1', 's2', 's3', 's4']

def generate_ml_classifiers(func):
    print(f'Training classifer based on {func}')
    # Load in data
    raw_training_data = pandas.read_pickle(
        f'persistence_statistics/training_data_embed_dim_3_pers_stats_{func}.pkl')
    raw_training_data = raw_training_data.dropna()

    # Make data wake = 1 and sleep = 0
    raw_training_data.loc[raw_training_data['sleep_stage'] != 1, 'sleep_stage'] = 0
    raw_training_data = raw_training_data.iloc[:, 1:]   # remove patient column

    # Train classifier
    classifiers_arr = [None]
    all_wake = raw_training_data[raw_training_data['sleep_stage'] == 1]
    num_wake = len(all_wake)

    # Separate the sleep_stage column
    X_train = raw_training_data.iloc[:, 1:]
    y_train = raw_training_data['sleep_stage']

    # Train SVM
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf.fit(X_train, y_train)
    print(clf.class_weight_)

    # Save classifier
    classifiers_arr[0] = clf
    print(classifiers_arr)
    print(f'########## Done training Classifier ###########')
    # with open(f"ml_classifiers/{func}_svm_classifier", "wb") as output:
    #     pickle.dump(classifiers_arr, output)


if __name__ == '__main__':
    func_arr = ['rips', 'alpha', 'del_rips']
    for func in func_arr:
        generate_ml_classifiers(func)
