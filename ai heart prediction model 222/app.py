"""
Heart Disease Predictor — All ML algorithms implemented from scratch using NumPy.
No sklearn model classes are used. Only NumPy & Pandas for data handling.
"""

import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ─────────────────────────────────────────────────────────────
#  ML IMPLEMENTATIONS — ALL FROM SCRATCH
# ─────────────────────────────────────────────────────────────
# ============================================================


# ─────────────────────────────────────
# 1. LOGISTIC REGRESSION FROM SCRATCH
# ─────────────────────────────────────
class LogisticRegressionScratch:
    """
    Logistic Regression implemented entirely from scratch using Gradient Descent.

    THEORY:
    -------
    Given input features X (shape: m×n), we want to predict probability that
    a sample belongs to class 1 (heart disease present).

    Step 1 — Linear combination:
        z = X·W + b         (dot product of inputs with learned weights)

    Step 2 — Sigmoid Activation (squashes z to range (0, 1)):
        σ(z) = 1 / (1 + e^(−z))

    Step 3 — Cost Function (Binary Cross-Entropy):
        J(W, b) = −(1/m) · Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]

    Step 4 — Gradient Descent (update weights to minimize cost):
        ∂J/∂W = (1/m) · Xᵀ · (ŷ − y)
        ∂J/∂b = (1/m) · Σ(ŷ − y)
        W  ←  W − α · ∂J/∂W
        b  ←  b − α · ∂J/∂b

    where α = learning rate, m = number of samples.
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations
        self.weights       = None   # shape: (n_features,)
        self.bias          = None   # scalar
        self.loss_history  = []     # track loss per iteration

    # ── Sigmoid ──────────────────────────────────────────────
    def _sigmoid(self, z):
        """Map any real value to (0, 1) using the sigmoid function."""
        z = np.clip(z, -500, 500)          # prevent numerical overflow in exp()
        return 1.0 / (1.0 + np.exp(-z))

    # ── Binary Cross-Entropy Loss ─────────────────────────────
    def _binary_cross_entropy(self, y_true, y_hat):
        """Compute binary cross-entropy loss."""
        m     = len(y_true)
        y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9)    # avoid log(0)
        return -(1 / m) * np.sum(
            y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)
        )

    # ── Training (Gradient Descent) ───────────────────────────
    def fit(self, X, y):
        """
        Train logistic regression weights using gradient descent.

        Parameters
        ----------
        X : ndarray, shape (m, n)  — Training features (already scaled)
        y : ndarray, shape (m,)    — Binary labels (0 or 1)
        """
        m, n = X.shape
        self.weights      = np.zeros(n)     # initialise weights to zero
        self.bias         = 0.0
        self.loss_history = []

        for iteration in range(self.n_iterations):
            # ── FORWARD PASS ──────────────────────────────────
            z     = np.dot(X, self.weights) + self.bias   # linear combination
            y_hat = self._sigmoid(z)                        # predicted probabilities

            # ── COMPUTE LOSS ──────────────────────────────────
            loss = self._binary_cross_entropy(y, y_hat)
            self.loss_history.append(loss)

            # ── BACKWARD PASS (calculate gradients) ───────────
            error = y_hat - y                              # prediction error
            dW    = (1 / m) * np.dot(X.T, error)          # gradient w.r.t weights
            db    = (1 / m) * np.sum(error)               # gradient w.r.t bias

            # ── PARAMETER UPDATE ──────────────────────────────
            self.weights -= self.learning_rate * dW
            self.bias    -= self.learning_rate * db

        return self

    # ── Prediction ────────────────────────────────────────────
    def predict_proba(self, X):
        """Return [P(class=0), P(class=1)] for each sample."""
        z            = np.dot(X, self.weights) + self.bias
        prob_pos     = self._sigmoid(z)
        return np.column_stack([1 - prob_pos, prob_pos])

    def predict(self, X):
        """Predict class label (0 or 1) using threshold = 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ──────────────────────────────────────
# 2. K-NEAREST NEIGHBORS FROM SCRATCH
# ──────────────────────────────────────
class KNearestNeighborsScratch:
    """
    K-Nearest Neighbors Classifier implemented from scratch.

    THEORY:
    -------
    KNN is a "lazy learner" — it does NOT build an explicit model during training.
    Instead it memorises the training data, then at prediction time:

    Step 1 — Compute Euclidean distance from query point x to every training point xᵢ:
        d(x, xᵢ) = √[ Σⱼ (xⱼ − xᵢⱼ)² ]

    Step 2 — Find K training points with smallest distances (the "K nearest neighbours").

    Step 3 — Majority vote: the most common class among K neighbours is the prediction.

    Step 4 — Probability: fraction of positive neighbours out of K.
    """

    def __init__(self, k=7):
        self.k        = k
        self.X_train  = None
        self.y_train  = None

    # ── Training (lazy — just store data) ─────────────────────
    def fit(self, X, y):
        """Store training data. KNN does no computation here."""
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=int)
        return self

    # ── Euclidean Distance ─────────────────────────────────────
    def _euclidean_distance(self, a, b):
        """d(a, b) = sqrt( sum( (aᵢ − bᵢ)² ) )"""
        return np.sqrt(np.sum((a - b) ** 2))

    # ── Find K Nearest labels ──────────────────────────────────
    def _get_k_nearest_labels(self, x):
        """
        Compute distance from x to all training points,
        sort them, and return the labels of the K nearest.
        """
        distances = np.array([
            self._euclidean_distance(x, x_train)
            for x_train in self.X_train
        ])
        k_indices = np.argsort(distances)[: self.k]     # indices of K smallest
        return self.y_train[k_indices]                   # their labels

    # ── Prediction ────────────────────────────────────────────
    def predict(self, X):
        """Predict class via majority vote for each sample in X."""
        predictions = []
        for x in np.array(X, dtype=float):
            k_labels    = self._get_k_nearest_labels(x)
            most_common = np.bincount(k_labels).argmax() # majority vote
            predictions.append(most_common)
        return np.array(predictions)

    def predict_proba(self, X):
        """Probability = fraction of positive labels among K neighbours."""
        probas = []
        for x in np.array(X, dtype=float):
            k_labels  = self._get_k_nearest_labels(x)
            pos_ratio = np.mean(k_labels)               # fraction that are class 1
            probas.append([1 - pos_ratio, pos_ratio])
        return np.array(probas)


# ─────────────────────────────────────────────────────────────
# 3. RANDOM FOREST FROM SCRATCH
#    (built on top of a Decision Tree implemented from scratch)
# ─────────────────────────────────────────────────────────────

# ── 3a. Internal Node structure ───────────────────────────────
class _TreeNode:
    """Represents one node inside a Decision Tree."""

    __slots__ = ["feature_idx", "threshold", "left", "right", "leaf_label"]

    def __init__(self):
        self.feature_idx = None   # which feature to split on
        self.threshold   = None   # split threshold value
        self.left        = None   # left child node  (X[:, feature] ≤ threshold)
        self.right       = None   # right child node (X[:, feature] >  threshold)
        self.leaf_label  = None   # only set for leaf nodes

    def is_leaf(self):
        return self.leaf_label is not None


# ── 3b. Decision Tree implemented from scratch ────────────────
class DecisionTreeScratch:
    """
    Binary Decision Tree (CART-style) built from scratch.

    THEORY:
    -------
    At each node we find the feature + threshold that best separates the classes
    by maximising Information Gain:

        IG = Entropy(parent) − [ (n_L/n)·Entropy(left) + (n_R/n)·Entropy(right) ]

    Entropy of a set S:
        H(S) = −Σ p_c · log₂(p_c)      (p_c = fraction of class c in S)

    We grow the tree recursively until a stopping criterion is met:
        • maximum depth reached
        • node is pure (only one class)
        • too few samples to split further

    Parameters
    ----------
    max_depth         : maximum allowed depth of the tree
    min_samples_leaf  : minimum samples required to form a leaf
    n_features_split  : how many features to consider at each split
                        (set to sqrt(total_features) by Random Forest for diversity)
    """

    def __init__(self, max_depth=8, min_samples_leaf=2, n_features_split=None):
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_features_split = n_features_split
        self.root             = None

    # ── Entropy ───────────────────────────────────────────────
    def _entropy(self, y):
        """H(y) = −Σ p_c · log₂(p_c)"""
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs  = counts / len(y)
        # add tiny epsilon to avoid log(0)
        return -np.sum([p * np.log2(p + 1e-9) for p in probs if p > 0])

    # ── Information Gain ──────────────────────────────────────
    def _information_gain(self, y_parent, feature_col, threshold):
        """
        IG = H(parent) − weighted_avg_entropy(children)

        Splits feature_col into left (≤ threshold) and right (> threshold).
        """
        left_mask  = feature_col <= threshold
        right_mask = ~left_mask

        n_parent = len(y_parent)
        n_left   = left_mask.sum()
        n_right  = right_mask.sum()

        if n_left == 0 or n_right == 0:
            return 0.0   # no real split

        H_parent   = self._entropy(y_parent)
        H_children = (
            (n_left  / n_parent) * self._entropy(y_parent[left_mask])  +
            (n_right / n_parent) * self._entropy(y_parent[right_mask])
        )
        return H_parent - H_children

    # ── Best Split Search ─────────────────────────────────────
    def _best_split(self, X, y):
        """
        Iterate over (feature, threshold) pairs and return the combo
        that gives the highest information gain.

        For speed, candidate thresholds are sampled via percentiles
        instead of trying every unique value.
        """
        n_features    = X.shape[1]
        n_feats_try   = self.n_features_split or n_features
        # pick a random subset of features (critical for Random Forest diversity)
        feature_ids   = np.random.choice(n_features, min(n_feats_try, n_features), replace=False)

        best_gain      = -1.0
        best_feature   = None
        best_threshold = None

        for feat_idx in feature_ids:
            col           = X[:, feat_idx]
            unique_vals   = np.unique(col)

            # Sample at most 20 candidate thresholds via percentiles for speed
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

    # ── Tree Growing (Recursive) ──────────────────────────────
    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the tree by finding best splits."""
        n_samples   = len(y)
        n_classes   = len(np.unique(y))
        majority    = int(np.bincount(y.astype(int)).argmax())

        # ── Stopping criteria → make a leaf ──────────────────
        if (
            depth >= self.max_depth
            or n_classes == 1
            or n_samples < self.min_samples_leaf * 2
        ):
            leaf = _TreeNode()
            leaf.leaf_label = majority
            return leaf

        # ── Find best split ───────────────────────────────────
        best_feat, best_thresh = self._best_split(X, y)

        if best_feat is None:
            leaf = _TreeNode()
            leaf.leaf_label = majority
            return leaf

        left_mask  = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        # If split produces an empty branch → leaf
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            leaf = _TreeNode()
            leaf.leaf_label = majority
            return leaf

        # ── Build internal node ───────────────────────────────
        node             = _TreeNode()
        node.feature_idx = best_feat
        node.threshold   = best_thresh
        node.left        = self._grow_tree(X[left_mask],  y[left_mask],  depth + 1)
        node.right       = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    # ── Fit ───────────────────────────────────────────────────
    def fit(self, X, y):
        self.root = self._grow_tree(np.array(X), np.array(y, dtype=int))
        return self

    # ── Single-sample traversal ───────────────────────────────
    def _traverse(self, x, node):
        """Walk one sample down the tree until a leaf is reached."""
        if node.is_leaf():
            return node.leaf_label
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    # ── Prediction ────────────────────────────────────────────
    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in np.array(X)])


# ── 3c. Random Forest (Bagging of Decision Trees) ─────────────
class RandomForestScratch:
    """
    Random Forest implemented from scratch via bagging of Decision Trees.

    THEORY:
    -------
    A single Decision Tree overfits — it memorises the training data.
    Random Forest reduces variance by combining many de-correlated trees:

    Step 1 — Bootstrap Sampling (Bagging):
        For each of the T trees, draw a random sample WITH replacement
        from the training data (same size as original). Each sample is
        called a "bootstrap sample". Some rows appear multiple times,
        some not at all (~36% are absent — these become the "Out-Of-Bag" set).

    Step 2 — Feature Bagging (Random Subspace):
        At every node split, consider only a random subset of
        √(n_features) features instead of all features.
        This de-correlates the trees, making their errors independent.

    Step 3 — Grow Decision Trees:
        Each tree is grown fully on its bootstrap sample using only
        the random feature subset at each split.

    Step 4 — Ensemble Prediction (Majority Vote):
        Aggregate predictions from all T trees → majority class wins.
        Probability = fraction of trees that predict class 1.

    Parameters
    ----------
    n_estimators    : number of trees in the forest
    max_depth       : max depth per tree
    min_samples_leaf: minimum leaf size per tree
    random_state    : seed for reproducibility
    """

    def __init__(self, n_estimators=25, max_depth=7, min_samples_leaf=2, random_state=42):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state
        self.trees            = []

    def fit(self, X, y):
        """Train the ensemble of Decision Trees on bootstrap samples."""
        np.random.seed(self.random_state)
        self.trees = []

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        n_samples, n_features = X.shape
        n_feats_per_split     = max(1, int(np.sqrt(n_features)))  # √n rule

        for tree_idx in range(self.n_estimators):
            # ── Step 1: Bootstrap sample ──────────────────────
            bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot        = X[bootstrap_idx]
            y_boot        = y[bootstrap_idx]

            # ── Step 2 & 3: Grow one Decision Tree ───────────
            tree = DecisionTreeScratch(
                max_depth        = self.max_depth,
                min_samples_leaf = self.min_samples_leaf,
                n_features_split = n_feats_per_split  # feature bagging
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X):
        """
        Step 4: Majority vote across all trees.
        tree_preds shape: (n_estimators, n_samples)
        """
        X          = np.array(X, dtype=float)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([
            np.bincount(tree_preds[:, i]).argmax()
            for i in range(X.shape[0])
        ])

    def predict_proba(self, X):
        """Probability = fraction of trees voting class 1."""
        X          = np.array(X, dtype=float)
        tree_preds = np.array([tree.predict(X) for tree in self.trees], dtype=float)
        pos_ratio  = tree_preds.mean(axis=0)          # average vote for class 1
        return np.column_stack([1 - pos_ratio, pos_ratio])


# ============================================================
# ─────────────────────────────────────────────────────────────
#  HELPER UTILITIES — all from scratch, no sklearn
# ─────────────────────────────────────────────────────────────
# ============================================================

# ── Standard Scaler from scratch ──────────────────────────────
def standard_scale(X_train, X_test=None):
    """
    Z-score normalisation (StandardScaler) from scratch:
        z = (x − μ) / σ
    Computed on training data; same μ and σ applied to test data.
    """
    mu    = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8   # avoid division by zero
    X_train_scaled = (X_train - mu) / sigma
    if X_test is not None:
        X_test_scaled = (X_test - mu) / sigma
        return X_train_scaled, X_test_scaled, mu, sigma
    return X_train_scaled, mu, sigma


# ── Train/Test Split from scratch ─────────────────────────────
def train_test_split_scratch(X, y, test_size=0.2, random_state=42):
    """Randomly split data into train and test sets."""
    np.random.seed(random_state)
    n          = len(X)
    indices    = np.random.permutation(n)
    test_count = int(n * test_size)
    test_idx   = indices[:test_count]
    train_idx  = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ── Evaluation Metrics from scratch ───────────────────────────
def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics from scratch (no sklearn).

    Confusion matrix:
        TP: predicted 1, actual 1
        TN: predicted 0, actual 0
        FP: predicted 1, actual 0
        FN: predicted 0, actual 1

    Accuracy   = (TP + TN) / total
    Precision  = TP / (TP + FP)      ← of all predicted positive, how many are actually positive
    Recall     = TP / (TP + FN)      ← of all actual positives, how many did we catch
    F1 Score   = 2 · (Precision · Recall) / (Precision + Recall)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy  = (TP + TN) / len(y_true)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "Accuracy":  round(accuracy  * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall":    round(recall    * 100, 2),
        "F1 Score":  round(f1        * 100, 2),
        "_cm": {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
    }


# ============================================================
# ─────────────────────────────────────────────────────────────
#  DATA LOADING & MODEL TRAINING
# ─────────────────────────────────────────────────────────────
# ============================================================

COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

FEATURE_NAMES = COLUMN_NAMES[:-1]   # drop 'target'


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load Cleveland heart disease dataset and do basic preprocessing."""
    df = pd.read_csv(data_path, header=None, names=COLUMN_NAMES, na_values='?')

    # Fill missing values with median (only 'ca' and 'thal' have missing values)
    for col in ['ca', 'thal']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Convert multi-class target to binary: 0 = no disease, 1 = disease present
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    return df


@st.cache_resource
def load_and_train(data_path: str, model_name: str):
    """
    Load data, scale features, train the chosen algorithm, and evaluate on a
    held-out 20% test set. Result is cached so training only happens once.

    Returns
    -------
    model       : trained model object
    scale_mu    : feature means (for scaling new inputs)
    scale_sigma : feature std deviations
    metrics     : dict with accuracy, precision, recall, F1
    df          : full dataframe (for display)
    """
    df = load_dataset(data_path)

    X = df.drop('target', axis=1).values.astype(float)
    y = df['target'].values.astype(int)

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split_scratch(X, y, test_size=0.2)

    # Standardise features (z-score)
    X_train_sc, X_test_sc, mu, sigma = standard_scale(X_train, X_test)

    # ── Select & Instantiate Model ─────────────────────────
    if model_name == "logistic_regression":
        model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    elif model_name == "knn":
        model = KNearestNeighborsScratch(k=7)
    elif model_name == "random_forest":
        model = RandomForestScratch(n_estimators=25, max_depth=7, min_samples_leaf=2)
    else:
        return None, None, None, None, None

    # ── Train ─────────────────────────────────────────────
    model.fit(X_train_sc, y_train)

    # ── Evaluate on test set ───────────────────────────────
    y_pred   = model.predict(X_test_sc)
    metrics  = compute_metrics(y_test, y_pred)

    return model, mu, sigma, metrics, df


# ============================================================
# ─────────────────────────────────────────────────────────────
#  STREAMLIT USER INTERFACE
# ─────────────────────────────────────────────────────────────
# ============================================================

# ── Sidebar — Algorithm selector ──────────────────────────────
st.sidebar.header("🧠 Choose ML Algorithm")
model_label = st.sidebar.selectbox(
    "Algorithm",
    ("Random Forest", "Logistic Regression", "K-Nearest Neighbors"),
    help="All three algorithms are built entirely from scratch using NumPy."
)

MODEL_MAP = {
    "Random Forest":        "random_forest",
    "Logistic Regression":  "logistic_regression",
    "K-Nearest Neighbors":  "knn",
}

# ── Load & train (cached) ─────────────────────────────────────
with st.spinner(f"Training {model_label} model from scratch..."):
    model, scale_mu, scale_sigma, metrics, df = load_and_train(
        'heart_data.txt', MODEL_MAP[model_label]
    )

# ── Page Title ────────────────────────────────────────────────
st.title("❤️ Heart Disease Predictor")
st.markdown(
    f"Currently using **{model_label}** — implemented **entirely from scratch** "
    "with NumPy (no sklearn model classes). Dataset: Cleveland Heart Disease (303 patients)."
)

# ── Algorithm Info Badge ──────────────────────────────────────
algo_desc = {
    "Logistic Regression":  "Gradient Descent · Sigmoid activation · Binary Cross-Entropy loss · α=0.1 · 1000 iterations",
    "K-Nearest Neighbors":  "Euclidean Distance · K=7 neighbours · Majority Vote · Lazy Learner (no training phase)",
    "Random Forest":        "25 Decision Trees · Bootstrap Sampling (Bagging) · Feature Bagging (√n features/split) · Entropy Criterion",
}
st.info(f"🔬 **Algorithm details:** {algo_desc[model_label]}", icon="ℹ️")

# ─────────────────────────────────────────────────────────────
# SIDEBAR — Patient input controls
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🏥 Patient Medical Details")

age      = st.sidebar.slider("Age (years)",              29,  77,  54)
sex      = st.sidebar.radio("Sex",                       ("Male", "Female"))
cp       = st.sidebar.selectbox("Chest Pain Type",       (1, 2, 3, 4),
                                help="1=Typical Angina, 2=Atypical Angina, 3=Non-anginal, 4=Asymptomatic")
trestbps = st.sidebar.slider("Resting BP (mm Hg)",       94, 200, 132)
chol     = st.sidebar.slider("Cholesterol (mg/dl)",     126, 564, 246)
fbs      = st.sidebar.radio("Fasting Blood Sugar >120 mg/dl", ("No", "Yes"))
restecg  = st.sidebar.selectbox("Resting ECG",           (0, 1, 2),
                                 help="0=Normal, 1=ST-T wave abnormality, 2=LV hypertrophy")
thalach  = st.sidebar.slider("Max Heart Rate Achieved",  71, 202, 150)
exang    = st.sidebar.radio("Exercise-Induced Angina",   ("No", "Yes"))
oldpeak  = st.sidebar.slider("ST Depression (Oldpeak)",  0.0, 6.2, 1.0, step=0.1)
slope    = st.sidebar.selectbox("ST Slope",              (1, 2, 3),
                                help="1=Upsloping, 2=Flat, 3=Downsloping")
ca       = st.sidebar.selectbox("Major Vessels (CA)",    (0, 1, 2, 3))
thal     = st.sidebar.selectbox("Thalassemia (Thal)",    (3, 6, 7),
                                help="3=Normal, 6=Fixed Defect, 7=Reversible Defect")

# Build raw input vector
raw_input = np.array([[
    age,
    1 if sex == "Male" else 0,
    cp,
    trestbps,
    chol,
    1 if fbs == "Yes" else 0,
    restecg,
    thalach,
    1 if exang == "Yes" else 0,
    oldpeak,
    slope,
    ca,
    thal
]], dtype=float)

# Scale input with the same μ and σ used during training
input_scaled = (raw_input - scale_mu) / (scale_sigma + 1e-8)

# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT — Three columns
# ─────────────────────────────────────────────────────────────
col_input, col_pred, col_metrics = st.columns([1.1, 1.5, 1.2])

# ── Column 1: Patient input table ─────────────────────────────
with col_input:
    st.subheader("📋 Patient Input")
    input_labels = {
        "Age": age, "Sex": sex, "Chest Pain": cp,
        "Resting BP": trestbps, "Cholesterol": chol,
        "Fasting BS": fbs, "Resting ECG": restecg,
        "Max HR": thalach, "Exercise Angina": exang,
        "Oldpeak": oldpeak, "ST Slope": slope,
        "Major Vessels": ca, "Thalassemia": thal,
    }
    input_display = pd.DataFrame(
        input_labels.items(), columns=["Feature", "Value"]
    ).set_index("Feature")
    st.dataframe(input_display, use_container_width=True)

# ── Column 2: Prediction ──────────────────────────────────────
with col_pred:
    st.subheader("🔮 Prediction")

    if model is not None:
        if st.button("🫀 Run Prediction", type="primary", use_container_width=True):
            prediction = model.predict(input_scaled)
            proba      = model.predict_proba(input_scaled)[0]   # [P(0), P(1)]

            if prediction[0] == 1:
                st.error("## 💔 HIGH Risk of Heart Disease")
                st.markdown(
                    f"The model estimates a **{proba[1]*100:.1f}%** probability "
                    "of heart disease being present."
                )
            else:
                st.success("## ❤️ LOW Risk of Heart Disease")
                st.markdown(
                    f"The model estimates a **{proba[0]*100:.1f}%** probability "
                    "of the patient being healthy."
                )

            # Probability table
            st.markdown("**Class Probabilities:**")
            prob_df = pd.DataFrame({
                "Class":       ["No Heart Disease (0)", "Heart Disease (1)"],
                "Probability": [f"{proba[0]*100:.2f}%",  f"{proba[1]*100:.2f}%"]
            })
            st.table(prob_df)
    else:
        st.error("❌ Model could not be loaded. Check the dataset path.")

# ── Column 3: Model Performance ───────────────────────────────
with col_metrics:
    st.subheader("📊 Model Performance")
    st.caption("Evaluated on 20% hold-out test set")

    if metrics:
        cm      = metrics.pop("_cm")
        for metric_name, val in metrics.items():
            st.metric(label=metric_name, value=f"{val}%")
        metrics["_cm"] = cm   # restore

        # Confusion matrix display
        st.markdown("**Confusion Matrix (test set):**")
        cm_df = pd.DataFrame(
            [[cm["TP"], cm["FN"]], [cm["FP"], cm["TN"]]],
            index   = ["Actual: Disease", "Actual: Healthy"],
            columns = ["Pred: Disease",   "Pred: Healthy"]
        )
        st.dataframe(cm_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# HOW IT WORKS — expandable algorithm explanation
# ─────────────────────────────────────────────────────────────
st.markdown("---")

with st.expander(f"🔬 How **{model_label}** Works — Full Implementation Details"):

    if model_label == "Logistic Regression":
        st.markdown("""
### Logistic Regression — Implemented from Scratch

Logistic Regression learns a **linear decision boundary** in feature space,
then maps it through a sigmoid function to output a probability.

---

#### Step 1 — Linear Combination
For a sample with features x = [x₁, x₂, ..., xₙ], compute:
```
z = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b
  = W · X  +  b          (vectorised dot product)
```
`W` are the **weights** (learned during training), `b` is the **bias**.

#### Step 2 — Sigmoid Activation
Squash `z` to the range (0, 1) so it can be interpreted as a probability:
```
σ(z) = 1 / (1 + e^(−z))
```
- If z is very large  → σ(z) ≈ 1.0  → high probability of disease
- If z is very small  → σ(z) ≈ 0.0  → low probability of disease
- If z = 0            → σ(z) = 0.5  → uncertain

#### Step 3 — Binary Cross-Entropy Loss
Measures how wrong our predictions are:
```
J(W, b) = −(1/m) · Σᵢ [ yᵢ · log(ŷᵢ) + (1−yᵢ) · log(1−ŷᵢ) ]
```
- `m` = number of training samples
- `yᵢ` = actual label (0 or 1)
- `ŷᵢ` = predicted probability

#### Step 4 — Gradient Descent
Iteratively move weights in the direction that reduces the loss:
```
∂J/∂W = (1/m) · Xᵀ · (ŷ − y)
∂J/∂b = (1/m) · Σ(ŷ − y)

W ← W − α · ∂J/∂W
b ← b − α · ∂J/∂b
```
`α = 0.1` (learning rate), repeated for **1000 iterations**.

#### Final Prediction
- If σ(W·x + b) ≥ 0.5  →  predict **Heart Disease**
- If σ(W·x + b) < 0.5  →  predict **No Heart Disease**
        """)

    elif model_label == "K-Nearest Neighbors":
        st.markdown("""
### K-Nearest Neighbors (K=7) — Implemented from Scratch

KNN is a **non-parametric, lazy learner** — it builds no model during training.
All the computation happens at prediction time.

---

#### Training Phase
Simply **store** all training samples and their labels in memory.
No parameters are learned.

#### Prediction Phase (for a new patient)

**Step 1 — Compute Euclidean Distance** to every training sample:
```
d(x, xᵢ) = √[ Σⱼ (xⱼ − xᵢⱼ)² ]
```
This gives a distance score for each of the ~242 training patients.

**Step 2 — Find the K=7 Nearest Neighbours**

Sort distances in ascending order and pick the top 7.

**Step 3 — Majority Vote**

Count how many of the 7 nearest neighbours have `label = 1` (disease) vs `label = 0`.
The majority class is the prediction.

**Step 4 — Probability**
```
P(Heart Disease) = (number of neighbours with label=1) / K
```

#### Why K=7?
Small K (like K=1) overfits to noise.
Large K smooths too much and underfits.
K=7 is a common empirically-tuned value for this dataset.
        """)

    elif model_label == "Random Forest":
        st.markdown("""
### Random Forest (25 Trees) — Implemented from Scratch

Random Forest is an **ensemble method** that trains many Decision Trees and
combines their votes to make a final prediction. It dramatically reduces the
overfitting problem of a single Decision Tree.

---

#### Building Block: Decision Tree (CART)

Each node finds the best feature + threshold by maximising **Information Gain**:

**Entropy** of a set S:
```
H(S) = −Σ p_c · log₂(p_c)
```
where `p_c` is the fraction of class `c` in set S.

**Information Gain** from splitting S on feature f at threshold t:
```
IG(S, f, t) = H(S) − [ (n_L/n)·H(S_left) + (n_R/n)·H(S_right) ]
```
The split with the **highest IG** is chosen.

---

#### Random Forest Training (for each of 25 trees)

**Step 1 — Bootstrap Sampling (Bagging)**
```
Draw n samples WITH replacement from training data
→ each tree sees a different (overlapping) view of the data
→ ~63% unique rows per tree, ~37% repeated
```

**Step 2 — Feature Bagging**
```
At every node split, randomly select  √n_features = √13 ≈ 4  features to consider
→ prevents all trees from splitting on the same dominant feature
→ makes each tree unique and uncorrelated
```

**Step 3 — Grow the Tree**
```
Recursively split until:
  • max_depth = 7 is reached, OR
  • node is pure (only one class), OR
  • too few samples remain
```

#### Prediction (Ensemble Vote)
```
Collect predictions from all 25 trees:
  tree_1 → 1, tree_2 → 0, tree_3 → 1, ..., tree_25 → 1

Majority vote:  15 trees say "1"  →  final prediction = Heart Disease
Probability  =  15 / 25 = 0.60   →  60% chance of heart disease
```

Why does this work? Each tree makes different errors (because of random sampling).
**Averaging many uncorrelated, slightly-wrong models → one highly accurate model.**
        """)

# ── Glossary ──────────────────────────────────────────────────
with st.expander("📖 Glossary of Medical Features"):
    st.markdown("""
| Feature | Full Name | Description |
|---|---|---|
| **Age** | Age | Patient age in years |
| **Sex** | Sex | 1 = Male, 0 = Female |
| **CP** | Chest Pain Type | 1=Typical Angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic |
| **Trestbps** | Resting Blood Pressure | Resting BP in mm Hg on hospital admission |
| **Chol** | Serum Cholesterol | Cholesterol in mg/dl |
| **FBS** | Fasting Blood Sugar | 1 = FBS > 120 mg/dl (True), 0 = False |
| **Restecg** | Resting ECG | 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy |
| **Thalach** | Max Heart Rate | Maximum heart rate achieved during stress test |
| **Exang** | Exercise Angina | Exercise-induced angina: 1=Yes, 0=No |
| **Oldpeak** | ST Depression | ST depression induced by exercise relative to rest |
| **Slope** | ST Slope | 1=Upsloping, 2=Flat, 3=Downsloping |
| **CA** | Major Vessels | Number of major vessels (0–3) coloured by fluoroscopy |
| **Thal** | Thalassemia | 3=Normal, 6=Fixed defect, 7=Reversible defect |
    """)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This tool is for **educational purposes only** and is NOT intended "
    "for clinical diagnosis. Always consult a qualified medical professional."
)
