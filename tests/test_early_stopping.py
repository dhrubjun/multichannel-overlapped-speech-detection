
import unittest

from src.training.early_stopping import EarlyStopping


class TestEarlyStopping(unittest.TestCase):

    def test_initial_score(self):
        stopper = EarlyStopping(patience=3)

        improved, stop = stopper.update(0.10, epoch=1)

        self.assertTrue(improved)
        self.assertFalse(stop)
        self.assertEqual(stopper.best_epoch, 1)

    def test_improvement_resets_patience(self):
        stopper = EarlyStopping(patience=2)

        stopper.update(0.10, 1)
        stopper.update(0.09, 2)

        improved, stop = stopper.update(0.12, 3)

        self.assertTrue(improved)
        self.assertFalse(stop)
        self.assertEqual(stopper.bad_epochs, 0)
        self.assertEqual(stopper.best_epoch, 3)

    def test_stops_after_patience(self):
        stopper = EarlyStopping(patience=2)

        stopper.update(0.20, 1)

        improved, stop = stopper.update(0.19, 2)
        self.assertFalse(improved)
        self.assertFalse(stop)

        improved, stop = stopper.update(0.18, 3)
        self.assertFalse(improved)
        self.assertTrue(stop)

    def test_min_delta(self):
        stopper = EarlyStopping(
            patience=2,
            min_delta=0.01,
        )

        stopper.update(0.20, 1)

        improved, stop = stopper.update(0.205, 2)

        self.assertFalse(improved)
        self.assertFalse(stop)

    def test_invalid_patience(self):
        with self.assertRaises(ValueError):
            EarlyStopping(patience=0)

    def test_nonfinite_score(self):
        stopper = EarlyStopping()

        with self.assertRaises(ValueError):
            stopper.update(float("nan"), 1)


if __name__ == "__main__":
    unittest.main()