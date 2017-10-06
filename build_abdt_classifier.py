import build_classifier as bc
import numpy as np

if __name__ == "__main__":

    num_max_depth = range(1, 102, 10)
    num_dts_range = range(20, 161, 20)
    learn_rate = np.arange(0.2, 1.1, 0.2)

    bc.train_test(
        bc.build_abdt(),
        {'max_depth': num_max_depth, 'n_estimators': num_dts_range, 'learning_rate': learn_rate},
        n_jobs_cv=20
    )