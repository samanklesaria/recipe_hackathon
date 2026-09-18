import csv

import numpy as np

import baseline_eval as be


def test_every_store_column_has_a_description():
    names = [c for c in next(csv.reader(open(be.PAIRS))) if c != "ingredient"]
    stores = be.load_stores(be.STORES, names)
    assert set(stores) == set(names)
    # The one entry that separates name from description with a period.
    assert stores["Seward Coop"].startswith("High priced food co-op")
    assert all("\n" not in d and len(d) > 20 for d in stores.values())


def test_calibrate_matches_the_true_label_frequencies():
    # 0 = always, 2 = sometimes, 3 = never.
    truth = np.array([[0, 0], [3, 2]])
    pred = be.calibrate(np.array([[0.1, 0.9], [0.5, 0.7]]), truth)
    # Ranked 0.9 > 0.7 > 0.5 > 0.1, so the two alwayses go to the top two.
    assert pred.tolist() == [[3, 0], [2, 0]]
    assert sorted(pred.ravel()) == sorted(truth.ravel())


def test_auc_is_1_when_scores_order_perfectly():
    pos = np.array([True, True, False, False])
    assert be.auc(np.array([0.9, 0.8, 0.2, 0.1]), pos) == 1.0
    assert be.auc(np.array([0.1, 0.2, 0.8, 0.9]), pos) == 0.0
    # Shape is irrelevant -- the matrix is ravelled either way.
    assert be.auc(np.array([[0.9, 0.8], [0.2, 0.1]]), pos.reshape(2, 2)) == 1.0
