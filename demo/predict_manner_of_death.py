from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from preprocess_wikidata import Preprocessor


def prfs_scoring(y_true, y_pred, beta=1.0, labels=None,
         pos_label=1, average=None,
         warn_for=('precision', 'recall', 'f-score'),
         sample_weight=None, metric_idx=0):

    return metrics.precision_recall_fscore_support(
        y_true, y_pred, beta=beta, labels=labels,
        pos_label=pos_label, average=average,
        warn_for=warn_for, sample_weight=sample_weight)[metric_idx]


# load json data
X = Preprocessor.assemble_feature_to_matrix_from_file("json/pp_preprocessed.json")
Y = Preprocessor.assemble_labels_to_vector_from_file("json/pp_preprocessed.json")

# split training and test data set
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.1
)

# classify
rfc = RandomForestClassifier(class_weight='balanced_subsample')
params = {
    'max_depth': range(5, 151, 2),
    'n_estimators': range(10, 151, 20)
}

# abdt = AdaBoostDecisionTrees(algorithm="SAMME")
# params = {
#    'max_depth': range(2, 4),
#    'n_estimators': [500],
#    'learning_rate': pl.frange(1.0, 1.0, 0.4)
#}

# tune hyper parameters, do k-fold cross validation to find the best hyper parameters
fscore = metrics.make_scorer(prfs_scoring, average='macro', metric_idx=2)
gs = GridSearchCV(rfc, params, cv=5, scoring=fscore)
gs.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(gs.best_params_)
print("Grid scores on development set:")
means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print "%0.3f (+/-%0.03f) for %r" % (mean, std, params)

# write
if True:
    with open("results/training-performance.txt", "w") as f:
        for mean, std, params in zip(means, stds, gs.cv_results_['params']):
            f.write("%0.3f %0.03f %r" % (mean, std, params))
            f.write("\n")

# transform label to integer
le = preprocessing.LabelEncoder()
le.fit(['accident', 'homicide', 'natural causes', 'suicide'])
y_true = le.transform(y_test)

# make prediction on testing data set
y_pred_label = gs.predict(X_test)
y_pred = le.transform(y_pred_label)

# measure performance
metric = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
print le.classes_
for met in metric:
    print met

if True:
    with open("results/testing-performance.txt", "w") as f:
        for cl in le.classes_:
            f.write(cl)
            f.write(" ")
        f.write("\n")
        for met in metric:
            for mm in met:
                f.write(str(mm))
                f.write(" ")
            f.write("\n")



