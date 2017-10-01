from sys import argv
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
import pickle

from preprocess_wikidata import Preprocessor


def _prfs_scoring(y_true, y_pred, beta=1.0, labels=None,
                  pos_label=1, average=None,
                  warn_for=('precision', 'recall', 'f-score'),
                  sample_weight=None, metric_idx=0):
    return metrics.precision_recall_fscore_support(
        y_true, y_pred, beta=beta, labels=labels,
        pos_label=pos_label, average=average,
        warn_for=warn_for, sample_weight=sample_weight)[metric_idx]

class ClassifierTrainee:

    def __init__(self, estimator, hyper_params=None):
        self.estimator = estimator
        self.hyper_params = hyper_params
        self._model = None
        self._label_encoder = None

    def train(self, X_train, y_train, hyper_params=None,
              train_perform_data_path=None, model_path=None,
              n_jobs=10, cv=5):
        """
        Train estimator and tune hyper parameters to find the best
        estimator via k-fold cross validation. Update the estimator
        as the best estimator found by cross validation.

        Parameters
        ----------
        :param X_train: array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        :param y_train: array-like, shape = [n_samples] or [n_samples, n_output]
            Target relative to X for classification or regression.

        :param hyper_params: dict or list of dictionaries, optional
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.
            If specified, the hyper parameters will of the instance will be
            updated.
            If the hyper parameters are not specified in neither initializer
            nor this function, a ValueError will be raised.

        :param train_perform_data_path: string, optional
            The destination file path. If this parameter are specified,
            the training performance (F Sore of every possible combination of
            hyper parameters) will be written into this file as json format.

        :param model_path: string, optional
            The destination file path. If this parameter are specified,
            the best estimator will be written into this file as pks bin format.

        :param n_jobs: int, optional (default=10)
            The number of jobs to run in parallel.

        :param cv: int, cross-validation generator or an iterable, optional (default=5)
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
                - None, to use 3-fold cross validation,
                - integer, to specify the number of folds in a `(Stratified)KFold`,
                - An object to be used as a cross-validation generator.
                - An iterable yielding train, test splits.

            For integer/None inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, :class:`StratifiedKFold` is used. In all
            other cases, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.


        Return
        ----------
        :return: GridSearchCV
        """

        # update hyper parameters to be tuned
        if hyper_params is not None:
            self.hyper_params = hyper_params
        # test parameters
        if self.hyper_params is None:
            raise ValueError("Hyper parameters are not set!")

        # tune hyper parameters, do k-fold cross validation to find the best hyper parameters
        fscore = metrics.make_scorer(_prfs_scoring, average='macro', metric_idx=2)
        print "\n start training..."
        gs = GridSearchCV(self.estimator, self.hyper_params, cv=cv, scoring=fscore, n_jobs=n_jobs)
        gs.fit(X_train, y_train)
        self._model = gs.best_estimator_
        print "training done! \n"

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

        # save model
        if model_path is not None:
            self.save_model(model_path)

        # encode labels
        self._label_encoder = preprocessing.LabelEncoder()
        self._label_encoder.fit(y_train)

        return gs

    def test(self, X_test, y_test, test_perform_data_path=None):
        """
        Test estimator stored in this instance.

        Parameters
        ----------
        :param X_test: array-like, shape = [n_samples, n_features]
            Testing vector, where n_samples is the number of samples and
            n_features is the number of features.

        :param y_test: array-like, shape = [n_samples]
            Target relative to X for classification or regression.

        :param test_perform_data_path: string, optional
            The destination file path. If this parameter are specified,
            the testing performance (in terms of precision, recall, F-Measure, Support)
            will be written into this file as json format.

        Returns
        ----------
        :return: array-like, shape = [n_samples]
            Output of the estimator.
        """

        # transform label to integer
        y_true = self._label_encoder.transform(y_test)

        # make prediction on testing data set
        y_pred_label = self._model.predict(X_test)
        y_pred = self._label_encoder.transform(y_pred_label)

        # measure performance
        metric = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
        print "\n testing performance: "
        print self._label_encoder.classes_
        for met in metric:
            print met

        if test_perform_data_path is not None:
            test_perform_json = []
            with open(test_perform_data_path, "w") as f:
                idx = 0
                for cl in self._label_encoder.classes_:
                    test_perform_json.append({cl:
                                                  {'precision': metric[0][idx],
                                                   'recall': metric[1][idx],
                                                   'F': metric[2][idx],
                                                   'support': metric[3][idx]}
                                              })
                    idx += 1
                json.dump(test_perform_json, f)

        return y_pred_label

    def save_model(self, model_path):
        """
        Save the current estimator of the instance.

        Parameters
        ----------
        :param model_path: string
            The destination file path.
        """
        with open(model_path, 'wb') as f:
            pickle.dump(self._model, f)

    def get_model(self):
        """
        Get the current estimator of the instance.

        Returns
        ---------
        :return: estimator.
        """
        return self._model

    def set_model(self, model):
        self._model = model

    def predict(self, X):
        """
        Make a prediction.

        Parameters
        ----------
        :param X: array-like, shape = [n_samples, n_features]
            Input vector, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ---------
        :return: array-like, shape = [n_samples]
            Output relative to X of the estimator.
        """
        return self._model.predict(X)


def get_args():
    args = {}
    raw = list(argv)
    while raw:
        if raw[0][0] == '-':
            args[raw[0]] = raw[1]
        raw = raw[1:]
    return args


def build_train_test(build_estimator_func, params,
                     train_test_split_ratio=0.1):
    """

    :param build_estimator_func:
    :param params:
    :param train_test_split_ratio:
    :return:
    """

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

    # load json data
    X = Preprocessor.assemble_feature_to_matrix_from_file(src_data_path)
    y = Preprocessor.assemble_labels_to_vector_from_file(src_data_path)

    # split training and test data set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_test_split_ratio
    )

    gs = build_estimator_func(X_train, y_train, train_perform_data_path, model_path, *params)
    gs.test(X_test, y_test, test_perform_data_path)


def build_lr(X_train, y_train, train_perform_data_path, model_path,
             Cs,
             multi_class='ovr', solver='liblinear', penalty='l1',
             n_jobs_estimator=1, n_jobs_cv=20):
    """

    :param X_train:
    :param y_train:
    :param train_perform_data_path:
    :param model_path:
    :param Cs:
    :param multi_class:
    :param solver:
    :param penalty:
    :param n_jobs_estimator:
    :param n_jobs_cv:
    :return:
    """

    # initialize estimator
    estimator = LogisticRegression(class_weight='balanced', n_jobs=n_jobs_estimator,
                             penalty=penalty, solver=solver,
                             multi_class=multi_class)

    # hyper parameters
    params = {
        'C': Cs,
    }

    # train estimator
    ct = ClassifierTrainee(estimator, params)
    ct.train(X_train=X_train, y_train=y_train,
             train_perform_data_path=train_perform_data_path,
             model_path=model_path, n_jobs=n_jobs_cv)

    # return trainer
    return ct


def build_rf(X_train, y_train, train_perform_data_path, model_path,
             num_max_depth, num_dts_range,
             class_weight='balanced_subsample', n_jobs_estimator=10, n_jobs_cv=10):
    """

    :param X_train:
    :param y_train:
    :param train_perform_data_path:
    :param model_path:
    :param num_max_depth:
    :param num_dts_range:
    :param class_weight:
    :param n_jobs_estimator:
    :param n_jobs_cv:
    :return:
    """

    # initialize estimator
    estimator = RandomForestClassifier(class_weight=class_weight, n_jobs=n_jobs_estimator)

    # hyper parameters
    params = {
        'max_depth': num_max_depth,
        'n_estimators': num_dts_range
    }

    ct = ClassifierTrainee(estimator, params)
    ct.train(X_train=X_train, y_train=y_train,
             train_perform_data_path=train_perform_data_path,
             model_path=model_path, n_jobs=n_jobs_cv)

    return ct