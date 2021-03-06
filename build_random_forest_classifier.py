import build_classifier as bc

if __name__ == "__main__":
    num_max_depth = range(5, 6, 5)
    num_dts_range = range(20, 21, 20)

    bc.train_test(
        bc.build_rf(),
        {'max_depth': num_max_depth, 'n_estimators': num_dts_range}
    )

