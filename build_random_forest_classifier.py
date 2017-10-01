import build_classifier as bc

if __name__ == "__main__":
    num_max_depth = range(5, 151, 5)
    num_dts_range = range(20, 101, 20)

    bc.build_train_test(
        bc.build_rf, [num_max_depth, num_dts_range]
    )

