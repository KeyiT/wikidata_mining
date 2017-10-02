import numpy as np
import build_classifier as bc

if __name__ == "__main__":

    Cs = np.arange(0.1, 2.1, 0.1)

    bc.train_test(
        bc.build_lr(), {'C': Cs}
    )