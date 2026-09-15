"""
Heart Disease Predictor Engine
All ML algorithms implemented from scratch using NumPy.
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────
# 1. LOGISTIC REGRESSION FROM SCRATCH
# ─────────────────────────────────────
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations
        self.weights       = None
        self.bias          = None
        self.loss_history  = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _binary_cross_entropy(self, y_true, y_hat):
        m     = len(y_true)
        y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9)
        return -(1 / m) * np.sum(
            y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)
        )

    def fit(self, X, y):
        m, n = X.shape
        self.weights      = np.zeros(n)
        self.bias         = 0.0
        self.loss_history = []

        for iteration in range(self.n_iterations):
            z     = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(z)

            loss = self._binary_cross_entropy(y, y_hat)
            self.loss_history.append(loss)

            error = y_hat - y
            dW    = (1 / m) * np.dot(X.T, error)
            db    = (1 / m) * np.sum(error)

            self.weights -= self.learning_rate * dW
            self.bias    -= self.learning_rate * db
        return self

    def predict_proba(self, X):
        z            = np.dot(X, self.weights) + self.bias
        prob_pos     = self._sigmoid(z)
        return np.column_stack([1 - prob_pos, prob_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ──────────────────────────────────────
# 2. K-NEAREST NEIGHBORS FROM SCRATCH
# ──────────────────────────────────────
class KNearestNeighborsScratch:
    def __init__(self, k=7):
        self.k        = k
        self.X_train  = None
        self.y_train  = None

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=int)
        return self

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _get_k_nearest_labels(self, x):
        distances = np.array([
            self._euclidean_distance(x, x_train)
            for x_train in self.X_train
        ])
        k_indices = np.argsort(distances)[: self.k]
        return self.y_train[k_indices]

    def predict(self, X):
        predictions = []
        for x in np.array(X, dtype=float):
            k_labels    = self._get_k_nearest_labels(x)
            most_common = np.bincount(k_labels).argmax()
            predictions.append(most_common)
        return np.array(predictions)

    def predict_proba(self, X):
        probas = []
        for x in np.array(X, dtype=float):
            k_labels  = self._get_k_nearest_labels(x)
            pos_ratio = np.mean(k_labels)
            probas.append([1 - pos_ratio, pos_ratio])
        return np.array(probas)


# ─────────────────────────────────────────────────────────────
# 3. RANDOM FOREST FROM SCRATCH
# ─────────────────────────────────────────────────────────────
class _TreeNode:
    __slots__ = ["feature_idx", "threshold", "left", "right", "leaf_label"]
    def __init__(self):
        self.feature_idx = None
        self.threshold   = None
        self.left        = None
        self.right       = None
        self.leaf_label  = None

    def is_leaf(self):
        return self.leaf_label is not None

class DecisionTreeScratch:
    def __init__(self, max_depth=8, min_samples_leaf=2, n_features_split=None):
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_features_split = n_features_split
        self.root             = None

    def _entropy(self, y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs  = counts / len(y)
        return -np.sum([p * np.log2(p + 1e-9) for p in probs if p > 0])

    def _information_gain(self, y_parent, feature_col, threshold):
        left_mask  = feature_col <= threshold
        right_mask = ~left_mask
        n_parent = len(y_parent)
        n_left   = left_mask.sum()
        n_right  = right_mask.sum()
        if n_left == 0 or n_right == 0:
            return 0.0
        H_parent   = self._entropy(y_parent)
        H_children = (
            (n_left  / n_parent) * self._entropy(y_parent[left_mask])  +
            (n_right / n_parent) * self._entropy(y_parent[right_mask])
        )
        return H_parent - H_children

    def _best_split(self, X, y):
        n_features    = X.shape[1]
        n_feats_try   = self.n_features_split or n_features
        feature_ids   = np.random.choice(n_features, min(n_feats_try, n_features), replace=False)

        best_gain      = -1.0
        best_feature   = None
        best_threshold = None

        for feat_idx in feature_ids:
            col           = X[:, feat_idx]
            unique_vals   = np.unique(col)
            if len(unique_vals) > 20:
                thresholds = np.percentile(col, np.linspace(5, 95, 20))
            else:
                thresholds = unique_vals

            for thresh in thresholds:
                gain = self._information_gain(y, col, thresh)
                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feat_idx
                    best_threshold = thresh
        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        n_samples   = len(y)
        n_classes   = len(np.unique(y))
        majority    = int(np.bincount(y.astype(int)).argmax())

        if (depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_leaf * 2):
            leaf = _TreeNode()
            leaf.leaf_label = majority
            return leaf

        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            leaf = _TreeNode()
            leaf.leaf_label = majority
            return leaf

        left_mask  = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            leaf = _TreeNode()
            leaf.leaf_label = majority
            return leaf

        node             = _TreeNode()
        node.feature_idx = best_feat
        node.threshold   = best_thresh
        node.left        = self._grow_tree(X[left_mask],  y[left_mask],  depth + 1)
        node.right       = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self._grow_tree(np.array(X), np.array(y, dtype=int))
        return self

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.leaf_label
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in np.array(X)])

class RandomForestScratch:
    def __init__(self, n_estimators=25, max_depth=7, min_samples_leaf=2, random_state=42):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state
        self.trees            = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        n_samples, n_features = X.shape
        n_feats_per_split     = max(1, int(np.sqrt(n_features)))

        for tree_idx in range(self.n_estimators):
            bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot        = X[bootstrap_idx]
            y_boot        = y[bootstrap_idx]
            tree = DecisionTreeScratch(
                max_depth        = self.max_depth,
                min_samples_leaf = self.min_samples_leaf,
                n_features_split = n_feats_per_split
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X          = np.array(X, dtype=float)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(tree_preds[:, i]).argmax() for i in range(X.shape[0])])

    def predict_proba(self, X):
        X          = np.array(X, dtype=float)
        tree_preds = np.array([tree.predict(X) for tree in self.trees], dtype=float)
        pos_ratio  = tree_preds.mean(axis=0)
        return np.column_stack([1 - pos_ratio, pos_ratio])


# ─────────────────────────────────────────────────────────────
#  HELPER UTILITIES
# ─────────────────────────────────────────────────────────────
def standard_scale(X_train, X_test=None):
    mu    = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    X_train_scaled = (X_train - mu) / sigma
    if X_test is not None:
        X_test_scaled = (X_test - mu) / sigma
        return X_train_scaled, X_test_scaled, mu, sigma
    return X_train_scaled, mu, sigma


def train_test_split_scratch(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n          = len(X)
    indices    = np.random.permutation(n)
    test_count = int(n * test_size)
    test_idx   = indices[:test_count]
    train_idx  = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

def load_dataset(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, header=None, names=COLUMN_NAMES, na_values='?')
    for col in ['ca', 'thal']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    return df

class HeartDiseaseModelEngine:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = RandomForestScratch() # Default model
        self.scale_mu = None
        self.scale_sigma = None
        self._train_model()

    def _train_model(self):
        df = load_dataset(self.data_path)
        X = df.drop('target', axis=1).values.astype(float)
        y = df['target'].values.astype(int)
        
        # We don't need train/test split for deployment inference, 
        # but we'll train on the full dataset for maximum accuracy.
        X_scaled, self.scale_mu, self.scale_sigma = standard_scale(X)
        self.model.fit(X_scaled, y)

    def predict(self, patient_features: list):
        """
        Expects a list of 13 features in order:
        ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        """
        input_arr = np.array([patient_features], dtype=float)
        input_scaled = (input_arr - self.scale_mu) / (self.scale_sigma + 1e-8)
        
        proba = self.model.predict_proba(input_scaled)[0]
        risk_percent = proba[1] * 100
        return risk_percent
