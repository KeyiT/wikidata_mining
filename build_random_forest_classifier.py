from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import json
from sys import argv
import pickle

from preprocess_wikidata import Preprocessor

def prfs_scoring(y_true, y_pred, beta=1.0, labels=None,
         pos_label=1, average=None,
         warn_for=('precision', 'recall', 'f-score'),
         sample_weight=None, metric_idx=0):

    return metrics.precision_recall_fscore_support(
        y_true, y_pred, beta=beta, labels=labels,
        pos_label=pos_label, average=average,
        warn_for=warn_for, sample_weight=sample_weight)[metric_idx]


def get_args():
    args = {}
    raw = list(argv)
    while raw:
        if raw[0][0] == '-':
            args[raw[0]] = raw[1]
        raw = raw[1:]
    return args


def train(X_train, y_train, train_perform_data_path, num_max_depth, num_dts_range, model_path):

    # classify
    rfc = RandomForestClassifier(class_weight='balanced_subsample', n_jobs=10)
    params = {
        'max_depth': num_max_depth,
        'n_estimators': num_dts_range
    }

    # tune hyper parameters, do k-fold cross validation to find the best hyper parameters
    fscore = metrics.make_scorer(prfs_scoring, average='macro', metric_idx=2)
    gs = GridSearchCV(rfc, params, cv=5, scoring=fscore, n_jobs=10)
    gs.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print(gs.best_params_)
    print("Grid scores on development set:")
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        print "%0.3f (+/-%0.03f) for %r" % (mean, std, params)

    # write training performance as json file
    if train_perform_data_path is not None:
        training_json = []
        for mean, std, params in zip(means, stds, gs.cv_results_['params']):
            res = dict(params)
            res['mean'] = mean
            res['std'] = std
            training_json.append(res)

        with open(train_perform_data_path, "w") as f:
            json.dump(training_json, f)

    if model_path is not None:
        save_model(gs.best_estimator_, model_path)

    return gs


def test(gs, X_test, y_test, test_perform_data_path, label_names):

    # transform label to integer
    le = preprocessing.LabelEncoder()
    le.fit(label_names)
    y_true = le.transform(y_test)

    # make prediction on testing data set
    y_pred_label = gs.predict(X_test)
    y_pred = le.transform(y_pred_label)

    # measure performance
    metric = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    print le.classes_
    for met in metric:
        print met

    if test_perform_data_path is not None:
        test_perform_json = []
        with open(test_perform_data_path, "w") as f:
            idx = 0
            for cl in le.classes_:
                test_perform_json.append({cl:
                                              {'precision': metric[0][idx],
                                               'recall': metric[1][idx],
                                               'F': metric[2][idx],
                                               'support': metric[3][idx]}
                                          })
                idx += 1
            json.dump(test_perform_json, f)

    return y_pred_label


def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def main():
    args = get_args()

    if '--src_json_data' in args:
        src_data_path = args['--src_json_data']
    else:
        raise ValueError("Source json data file path is empty")

    train_perform_data_path = None
    if '--train_perform' in args:
        train_perform_data_path = args['--train_perform']

    test_perform_data_path = None
    if '--test_perform' in args:
        test_perform_data_path = args['--test_perform']

    model_path = None
    if '--model_bin' in args:
        model_path = args['--model_bin']

    num_max_depth = range(5, 151, 5)
    num_dts_range = range(20, 101, 20)

    # load json data
    X = Preprocessor.assemble_feature_to_matrix_from_file(src_data_path)
    Y = Preprocessor.assemble_labels_to_vector_from_file(src_data_path)

    # split training and test data set
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1
    )

    gs = train(X_train, y_train, train_perform_data_path, num_max_depth, num_dts_range, model_path)
    test(gs, X_test, y_test, test_perform_data_path, Y)

if __name__ == "__main__":
    main()

