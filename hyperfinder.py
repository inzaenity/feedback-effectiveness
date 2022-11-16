
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import re


TFIDF_HYPERS = {
    "sublinear_tf": [True, False], 
    "stop_words": ["english", None], 
    "ngram_range": [(1, k) for k in range(1, 4)],
    "max_features": [None, 10000, 20000, 50000],
    "binary": [True, False],
    "norm": ["l1", "l2"],
    "use_idf": [True, False],
    "lowercase": [True, False],
}


# Given a learning method (e.g. LogisticRegression()) and a map of
# hyperparamter name to list of values to try, the HyperFinder finds
# the best set of hyperparameters for the learning method.
class HyperFinder:
    def __init__(self, method, hypers_space):
        self.algorithm = Pipeline([("tfidf", TfidfVectorizer()), ("algo", method)])
        self.hypers_space = hypers_space

    # Find the best hyperparameters and return a Pipeline with those
    # hyperparameters set
    def get_best_model(self, X, y, export_results=False, discourse_type=None):
        method_hypers = {"algo__" + hyper : self.hypers_space[hyper] for hyper in self.hypers_space}
        tfidf_hypers = {"tfidf__" + hyper : TFIDF_HYPERS[hyper] for hyper in TFIDF_HYPERS}
        clf = RandomizedSearchCV(self.algorithm, method_hypers | tfidf_hypers, n_iter=20, n_jobs=-1, random_state=10)
        search = clf.fit(X, y)
        best_hypers = search.best_params_
        
        if export_results:
            export_cv_results(clf.cv_results_, discourse_type)
        
        return self.algorithm.set_params(**best_hypers)


def export_cv_results(results, discourse_type):
    df = pd.DataFrame(results)
    cols = [name for name in list(df) if re.search("^param_|_score$", name) and re.search("^split", name) == None]
    df = df[cols]
    df = df.sort_values("rank_test_score")
    renaming = {name : name.split("__")[1] for name in list(df.columns) if "param" in name}
    renaming |= {"mean_test_score": "Mean test score", "std_test_score": "Standard deviation"}
    df.rename(columns=renaming, inplace=True)
    df.drop(columns="rank_test_score", inplace=True)
    df["Mean test score"] = df["Mean test score"].apply(lambda x: round(x, 3))
    df["Standard deviation"] = df["Standard deviation"].apply(lambda x: round(x, 3))
    df.to_csv(f"final_model_training/{discourse_type}.csv")

