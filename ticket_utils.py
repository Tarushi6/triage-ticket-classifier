"""
ticket_utils.py — shared code for train.py and app.py.

Keep these classes/functions in an importable module (NOT in train.py).
joblib pickles classes by their module path, so anything defined in a script
run as `python train.py` is stored as `__main__.X` and cannot be loaded by
app.py. Import from here in both places:

    from ticket_utils import clean_text, LabelEncodedClassifier
"""

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_ST = True
except ImportError:
    HAS_ST = False


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def clean_text(text) -> str:
    """Lowercase, strip URLs and punctuation, collapse whitespace.

    Digits are kept, and dots *between digits* are kept too, so versions like
    'vpn 2.1' survive as '2.1' instead of splitting into '2' and '1'.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s.]", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)  # drop dots not between digits
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Keeps tokens like '2.1' and single characters/digits ('a', '5').
TOKEN_PATTERN = r"(?u)\b\w[\w.]*\w\b|\b\w\b"


# ---------------------------------------------------------------------------
# Wrapper for XGBoost / CatBoost (they need integer labels)
# ---------------------------------------------------------------------------


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Encodes string labels to 0..n-1 on fit and decodes on predict.

    Mixin order (ClassifierMixin first) is required by scikit-learn >= 1.6.
    predict_proba columns follow LabelEncoder.classes_, so self.classes_
    lines up with them without reordering.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder()
        y_encoded = self.encoder_.fit_transform(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_encoded)
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(self.estimator_.predict(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# ---------------------------------------------------------------------------
# Sentence embeddings (optional: pip install sentence-transformers)
# ---------------------------------------------------------------------------

# text -> vector cache, so cross-validation and the second label (Priority)
# don't re-encode the same tickets. Not pickled with the model.
_EMB_CACHE = {}


class SentenceEmbedder(TransformerMixin, BaseEstimator):
    """Turns text into dense meaning vectors with a pretrained model.

    Stateless (nothing is learned from your data), so it cannot leak. The
    loaded model is dropped when pickling and reloaded on first use, which
    means app.py also needs sentence-transformers installed (the pretrained
    weights are downloaded once, then cached by the library).
    """

    def __init__(
        self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64
    ):
        self.model_name = model_name
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def _get_model(self):
        model = getattr(self, "_model", None)
        if model is None:
            if not HAS_ST:
                raise ImportError(
                    "sentence-transformers is required: pip install sentence-transformers"
                )
            model = SentenceTransformer(self.model_name)
            self._model = model
        return model

    def transform(self, X):
        texts = [str(t) for t in X]
        missing = [
            t for t in dict.fromkeys(texts) if (self.model_name, t) not in _EMB_CACHE
        ]
        if missing:
            if len(missing) > 1000:
                print(f"    encoding {len(missing)} texts with {self.model_name} ...")
            vecs = self._get_model().encode(
                missing,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(missing) > 1000,
            )
            for t, v in zip(missing, vecs):
                _EMB_CACHE[(self.model_name, t)] = np.asarray(v, dtype=np.float32)
        return np.vstack([_EMB_CACHE[(self.model_name, t)] for t in texts])

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_model", None)
        return state


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------


def word_tfidf(max_features: int, ngram_max: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, ngram_max),
        token_pattern=TOKEN_PATTERN,
        sublinear_tf=True,
    )


def hybrid_features(max_features: int, ngram_max: int) -> FeatureUnion:
    """Word n-grams + character n-grams side by side.

    Word features capture topic ('vpn', 'invoice'); character n-grams are
    robust to typos, odd product names and error codes the word vocabulary
    misses. Combining them usually beats either alone on short noisy text.
    """
    return FeatureUnion(
        [
            ("word", word_tfidf(max_features, ngram_max)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------

# Too slow to cross-validate by default (see --cv-slow in train.py).
NO_CV_MODELS = {
    "AdaBoost",
    "SVC (rbf kernel)",
    "SVC (poly kernel)",
    "GradientBoosting",
    "CatBoost",
}


def _base_learners():
    """Diverse, complementary learners used by both hybrid ensembles."""
    return [
        (
            "lr",
            LogisticRegression(max_iter=1000, class_weight="balanced", C=5),
        ),
        (
            "svc",
            CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
        ),
        ("cnb", ComplementNB(alpha=0.3)),
    ]


def _svc_stack(max_features: int, ngram_max: int, with_embeddings: bool):
    """SVC-centred stacking hybrid.

    Each base learner sees its OWN view of the raw text (its own vectorizer
    inside its own pipeline), so the views are genuinely different:
      * LinearSVC on word TF-IDF       - best raw accuracy in earlier runs
      * LogisticRegression on word TF-IDF - best macro-F1 in earlier runs
      * ComplementNB on word TF-IDF    - cheap, good on rare classes
      * (with --embeddings) RBF-kernel SVC (Nystroem approx) on sentence
        embeddings + LogisticRegression on embeddings - meaning-based views

    A meta-learner then learns how much to trust each model's scores. Base
    predictions come from internal cross-validation, so the meta-learner never
    sees a base model's score on data that model was trained on.
    stack_method="auto" uses predict_proba where it exists and the SVC's
    decision_function otherwise; the scores are standardised before the
    meta-learner because those two scales differ.
    """

    def word():
        return word_tfidf(max_features, ngram_max)

    estimators = [
        (
            "svc_word",
            Pipeline(
                [
                    ("t", word()),
                    ("c", LinearSVC(C=0.3, class_weight="balanced")),
                ]
            ),
        ),
        (
            "lr_word",
            Pipeline(
                [
                    ("t", word()),
                    (
                        "c",
                        LogisticRegression(
                            max_iter=1000, class_weight="balanced", C=10
                        ),
                    ),
                ]
            ),
        ),
        (
            "cnb_word",
            Pipeline([("t", word()), ("c", ComplementNB(alpha=0.3))]),
        ),
    ]
    if with_embeddings:
        estimators += [
            (
                "svc_emb",
                Pipeline(
                    [
                        ("e", SentenceEmbedder()),
                        (
                            "k",
                            Nystroem(
                                kernel="rbf",
                                gamma=1.0,
                                n_components=1000,
                                random_state=42,
                            ),
                        ),
                        ("c", LinearSVC(C=1.0, class_weight="balanced")),
                    ]
                ),
            ),
            (
                "lr_emb",
                Pipeline(
                    [
                        ("e", SentenceEmbedder()),
                        (
                            "c",
                            LogisticRegression(
                                max_iter=2000, class_weight="balanced", C=10
                            ),
                        ),
                    ]
                ),
            ),
        ]

    return StackingClassifier(
        estimators=estimators,
        final_estimator=make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)
        ),
        stack_method="auto",
        cv=3,
    )


def build_candidates(
    max_features: int,
    ngram_max: int,
    fast: bool = False,
    embeddings: bool = False,
    char_features: bool = False,
) -> dict:
    """Name -> unfitted Pipeline(tfidf -> classifier).

    Because the vectorizer lives INSIDE each pipeline, cross-validation and
    the train/test split never leak vocabulary or IDF weights from held-out
    rows. Fit and predict on raw (cleaned) text.
    """

    def pipe(clf, hybrid=False):
        feats = (
            hybrid_features(max_features, ngram_max)
            if (hybrid and char_features)
            else word_tfidf(max_features, ngram_max)
        )
        return Pipeline([("tfidf", feats), ("clf", clf)])

    candidates = {
        "MultinomialNB": pipe(MultinomialNB()),
        "ComplementNB": pipe(ComplementNB()),
        "LogisticRegression": pipe(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "LinearSVC (calibrated)": pipe(
            CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)
        ),
        "SVC (rbf, Nystroem approx)": pipe(
            make_pipeline(
                Nystroem(kernel="rbf", gamma=0.5, n_components=300, random_state=42),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            )
        ),
        "AdaBoost": pipe(AdaBoostClassifier(n_estimators=200, random_state=42)),
        # --- Regularisation (C) tuning of the strongest linear models --------
        # Smaller C = stronger regularisation, larger C = fits training data
        # harder. The defaults (C=1) are rarely optimal for TF-IDF.
        "LinearSVC C=0.3 (calibrated)": pipe(
            CalibratedClassifierCV(
                LinearSVC(class_weight="balanced", C=0.3), cv=3
            )
        ),
        "LinearSVC C=3 (calibrated)": pipe(
            CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=3), cv=3)
        ),
        "LogisticRegression C=10": pipe(
            LogisticRegression(max_iter=1000, class_weight="balanced", C=10)
        ),
        "LogisticRegression C=30": pipe(
            LogisticRegression(max_iter=1000, class_weight="balanced", C=30)
        ),
    }

    # --- Hybrid models -----------------------------------------------------
    # Soft voting uses word TF-IDF only by default (fast). --char adds
    # character n-grams (~10x slower per fit; did not help on this dataset).
    tag = "word+char" if char_features else "word"
    if char_features:
        candidates["Hybrid: word+char LR"] = pipe(
            LogisticRegression(max_iter=1000, class_weight="balanced", C=5),
            hybrid=True,
        )
    candidates[f"Hybrid: soft voting ({tag})"] = pipe(
        VotingClassifier(_base_learners(), voting="soft"), hybrid=True
    )

    # The main hybrid: SVC-centred stack (see _svc_stack). Takes raw text and
    # builds its own per-learner features, so it is NOT wrapped in pipe().
    emb_ok = embeddings and HAS_ST
    candidates[f"Hybrid: SVC stack ({'word+emb' if emb_ok else 'word'})"] = (
        _svc_stack(max_features, ngram_max, emb_ok)
    )

    if not fast:
        candidates["SVC (rbf kernel)"] = pipe(
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                cache_size=1000,
                random_state=42,
            )
        )
        candidates["SVC (poly kernel)"] = pipe(
            SVC(
                kernel="poly",
                degree=2,
                C=10,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                cache_size=1000,
                random_state=42,
            )
        )
        candidates["GradientBoosting"] = pipe(
            GradientBoostingClassifier(n_estimators=150, random_state=42)
        )

    if embeddings:
        if not HAS_ST:
            print(
                "Note: sentence-transformers not installed — skipping embedding "
                "models. (pip install sentence-transformers)"
            )
        else:
            # Meaning-based features: 'cannot log in' ~ 'login failing'.
            candidates["Embeddings + LR"] = Pipeline(
                [
                    ("emb", SentenceEmbedder()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000, class_weight="balanced", C=10
                        ),
                    ),
                ]
            )
            # Hybrid: sparse word TF-IDF (exact keywords) + dense embeddings
            # (meaning) into one calibrated linear SVM.
            candidates["Hybrid: tfidf+embeddings SVC"] = Pipeline(
                [
                    (
                        "feats",
                        FeatureUnion(
                            [
                                ("word", word_tfidf(max_features, ngram_max)),
                                ("emb", SentenceEmbedder()),
                            ]
                        ),
                    ),
                    (
                        "clf",
                        CalibratedClassifierCV(
                            LinearSVC(class_weight="balanced"), cv=3
                        ),
                    ),
                ]
            )

    if HAS_XGB:
        candidates["XGBoost"] = pipe(
            LabelEncodedClassifier(
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.2,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    eval_metric="mlogloss",
                    n_jobs=-1,
                    random_state=42,
                )
            )
        )

    if HAS_CATBOOST:
        candidates["CatBoost"] = pipe(
            LabelEncodedClassifier(
                CatBoostClassifier(
                    iterations=300,
                    depth=6,
                    learning_rate=0.2,
                    loss_function="MultiClass",
                    verbose=0,
                    allow_writing_files=False,
                    random_seed=42,
                )
            )
        )

    return candidates