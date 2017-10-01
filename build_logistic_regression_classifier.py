import numpy as np
import build_classifier as bc

if __name__ == "__main__":

    Cs = np.arange(0.2, 0.5, 0.2)
    bc.build_train_test(
        bc.build_lr, [Cs]
    )