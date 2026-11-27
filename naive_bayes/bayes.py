import numpy as np

class MultinomialNB:

    def __init__(self, alpha = 1.0):
        self.alpha = alpha  

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_words = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_words))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]

            self.class_log_prior_[i] = np.log(len(X_c) / len(X))

            word_counts = X_c.sum(axis=0)
            total = word_counts.sum()
            self.feature_log_prob_[i] = np.log((word_counts + self.alpha) /
                                               (total + self.alpha * n_words))

        return self

    def predict_log_scores(self, X):
      
        X = np.asarray(X)
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        scores = self.predict_log_scores(X)
        best = np.argmax(scores, axis=1)  
        return self.classes_[best]

    def predict_proba(self, X):
        scores = self.predict_log_scores(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / probs.sum(axis=1, keepdims=True)