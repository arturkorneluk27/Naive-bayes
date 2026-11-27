import unittest

import numpy as np

from bayes import MultinomialNB


class TestFit(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[2, 1, 0],
                           [1, 1, 0],
                           [0, 1, 3],
                           [0, 0, 2]])
        self.y = np.array([0, 0, 1, 1])

    def test_classes(self):
        model = MultinomialNB().fit(self.X, self.y)
        np.testing.assert_array_equal(model.classes_, [0, 1])

    def test_priors_sum_to_one(self):
        model = MultinomialNB().fit(self.X, self.y)
        self.assertAlmostEqual(np.exp(model.class_log_prior_).sum(), 1.0)

    def test_priors_values(self):
        model = MultinomialNB().fit(self.X, [0, 1, 1, 1])
        np.testing.assert_allclose(np.exp(model.class_log_prior_), [0.25, 0.75])

    def test_word_probs_sum_to_one(self):
        model = MultinomialNB().fit(self.X, self.y)
        sums = np.exp(model.feature_log_prob_).sum(axis=1)
        np.testing.assert_allclose(sums, [1.0, 1.0])

    def test_word_probs_by_hand(self):
        model = MultinomialNB(alpha=1.0).fit(self.X, self.y)
        np.testing.assert_allclose(np.exp(model.feature_log_prob_[0]), [4/8, 3/8, 1/8])

    def test_no_zero_probs(self):
        model = MultinomialNB().fit(self.X, self.y)
        self.assertTrue(np.all(np.isfinite(model.feature_log_prob_)))

    def test_three_classes(self):
        model = MultinomialNB().fit(self.X, [0, 1, 2, 2])
        self.assertEqual(model.feature_log_prob_.shape, (3, 3))


class TestPredict(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[2, 1, 0],
                           [1, 1, 0],
                           [0, 1, 3],
                           [0, 0, 2]])
        self.y = np.array([0, 0, 1, 1])
        self.model = MultinomialNB().fit(self.X, self.y)

    def test_predict_training_data(self):
        np.testing.assert_array_equal(self.model.predict(self.X), self.y)

    def test_predict_new_documents(self):
        X_new = np.array([[3, 0, 0],   
                          [0, 0, 5]])  
        np.testing.assert_array_equal(self.model.predict(X_new), [0, 1])

    def test_predict_text_labels(self):
        model = MultinomialNB().fit(self.X, ["ham", "ham", "spam", "spam"])
        self.assertEqual(model.predict([[0, 0, 4]])[0], "spam")

    def test_proba_sums_to_one(self):
        probs = self.model.predict_proba(self.X)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(4))

    def test_proba_matches_predict(self):
        probs = self.model.predict_proba(self.X)
        np.testing.assert_array_equal(np.argmax(probs, axis=1), self.model.predict(self.X))

    def test_long_document_no_underflow(self):
        probs = self.model.predict_proba([[1000, 1000, 1000]])
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertAlmostEqual(probs.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()