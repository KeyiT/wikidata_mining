import build_classifier as bc
from sklearn.feature_selection import SelectFromModel
import numpy as np



class MyFeatureSelector(bc.DataManipulator):

    def __init__(self, feature_selector):
        self.feature_selector = feature_selector

    def manipulate_data(self, X, y=None):
        """
        Select important features.

        :param X: matrix
            The feature matrix.

        :param y: list, optional
            The corresponding labels.

        :returns processed X and y
        """
        print "Original feature matrix shape:"
        print X.shape
        X = self.feature_selector.transform(X)
        print "Selected feature matrix shape:"
        print X.shape

        return X, y



def find_top_features_threshold(feat_import_, max_feat_import_portion):
    feature_tuples = []
    for idx in xrange(0, len(feat_import_)):
        feature_tuples.append((idx, feat_import_[idx]))

    feature_tuples.sort(key=lambda feature: feature[1], reverse=True)

    # pick top features
    picked_portion = 0
    threshold = 0
    for feat in feature_tuples:
        picked_portion += feat[1]
        if picked_portion > max_feat_import_portion:
            break

        threshold = feat[1]

    return threshold


if __name__ == "__main__":

    # train random forest for feature selection
    num_max_depth = range(83, 84)
    num_dts_range = range(80, 81)

    rf_trainer = bc.train_test(
        bc.build_rf(n_jobs_estimator=15),
        {'max_depth': num_max_depth, 'n_estimators': num_dts_range}
    )

    rf = rf_trainer.get_model()

    # select features
    th = find_top_features_threshold(rf.feature_importances_, 0.95)
    model = SelectFromModel(rf, prefit=True, threshold=th)

    selector = MyFeatureSelector(model)

    # train logistic regression
    Cs = np.arange(0.1, 2.1, 0.1)
    bc.train_test(
        bc.build_lr(), {'C': Cs}, data_manipulator=selector
    )
