# Author: Amish Mishra
# Date: June 6, 2022
# README: This file validates classifiers on the CGMH validation data, then it generates some model accuracy metrics

from sklearn import metrics
import pandas
import numpy as np
import pickle


def metrics_from_confusion_matrix(cm, y_test, y_pred_prob, verbose=True):
    '''
    This function returns the accuracy metrics for a given confusion matrix
    '''
    tn, fp, fn, tp = cm.ravel()
    precision = tp/(tp + fp)
    recall = tp/(tp+fn)
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    specificity = tn / (tn+fp)
    f1_score = tp/(tp + 0.5*(fp + fn))
    auc = metrics.roc_auc_score(y_test, y_pred_prob[:,1])
    aps = metrics.average_precision_score(y_test, y_pred_prob[:,1])
    kappa = (2*(tp*tn-fn*fp))/((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn))
    if verbose:
        print('Confusion matrix:\n ', cm)
        print('TP:', tp)
        print('FP:', fp)
        print('TN:', tn)
        print('FN:', fn)
        print('SE %:', recall*100)
        print('SP %:', specificity*100)
        print('Acc %:', accuracy*100)
        print('PR %:', precision*100)
        print('F1 score:', f1_score)
        print('AUC:', auc)
        print('APS:', aps)
        print('Kappa:', kappa)
    return [tp, fp, tn, fn, recall, specificity, 
            accuracy, precision, f1_score, auc, aps, kappa]


def validate_ml_classifiers(func):
    print(f'Validating {func} svm...')
    # For reference: stages = ['rem', 'wake', 's1', 's2', 's3', 's4']
    # Load in data and classfiers
    validation_data = pandas.read_pickle(
        f'persistence_statistics/validation_embed_dim_3_pers_stats_{func}.pkl')
    validation_data = validation_data.dropna()
    with open(f"ml_classifiers/{func}_svm_classifier", "rb") as data:
        classifier = pickle.load(data)

    # Make data wake = 1 and sleep = 0
    validation_data.loc[validation_data['sleep_stage'] != 1, 'sleep_stage'] = 0

    # Build an empty data frame of performance metrics for each patient
    num_patients = validation_data['patient'].drop_duplicates().count()
    perf_metrics = ['tp', 'fp', 'tn', 'fn', 'se', 'sp', 'acc', 'pr', 'f1', 'auc', 'aps', 'kappa']
    metrics_df = pandas.DataFrame(columns=np.arange(1, num_patients+1, 1), index=perf_metrics)

    for p in range(1, num_patients+1):
        # print(f'---------Patient {p}-----------')
        X = validation_data[validation_data['patient'] == p]
        X_test = X.iloc[:, 2:]
        y_test = X['sleep_stage']
        pred = classifier.predict(X_test)
        y_pred_prob = classifier.predict_proba(X_test)
        cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
        metrics_vect = metrics_from_confusion_matrix(cm, y_test, y_pred_prob, verbose=False)
        metrics_df[p] = metrics_vect
    print(metrics_df)
    target_location = f'performance_metrics_tables/perf_metrics_{func}_svm_classifier.pkl'
    metrics_df.to_pickle(target_location)
    print('Performance metrics table saved in', target_location)


if __name__ == '__main__':
    func_arr = ['rips', 'alpha', 'del_rips']
    for func in func_arr:
        validate_ml_classifiers(func)