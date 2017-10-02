import build_classifier as bc

if __name__ == "__main__":

    num_max_depth = range(2, 20, 2)
    num_dts_range = range(20, 101, 20)

    bc.train_test(
        bc.build_gbdt(),
        {'max_depth': num_max_depth, 'n_estimators': num_dts_range}
    )