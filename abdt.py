from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class AdaBoostDecisionTrees(AdaBoostClassifier):
    def __init__(self,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None,
                 max_depth=3):

        super(AdaBoostDecisionTrees, self).__init__(
            base_estimator=DecisionTreeClassifier(max_depth=max_depth),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state
        )