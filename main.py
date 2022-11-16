
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import time

from hyperfinder import HyperFinder


OUTER_CV_K = 5
TEST_SIZE = 0.2
NB_HYPERS = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]}
DT_HYPERS = {"min_samples_leaf": list(range(1, 31)), "criterion": ["gini", "entropy", "log_loss"]}
KNN_HYPERS = {"n_neighbors": list(range(1, 21)), "weights": ["uniform", "distance"]}
LR_HYPERS = {"C": [0.5] + list(range(1, 9)), "class_weight": ["balanced", None]}


# Return the best model for each discourse type, as a map from
# discourse type (string) to a model.
def train_models():
    df = pd.read_csv("data/train.csv")
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=7, stratify=df["discourse_effectiveness"])
    discourse_types = df["discourse_type"].dropna().unique()
    best_models = dict()
    final_results = {
        "Discourse type": list(discourse_types),
        "Best algorithm": [],
        "Test log loss": [],
    }
    for t in discourse_types:
        # For each discourse type, choose the best learning algorithm,
        # then train it on the entire training data
        print(f"[INFO] Training a model for \"{t}\"...")
        df_t = df_train[df_train["discourse_type"] == t]
        best_algo, best_loss, sd = choose_best_algorithm(df_t, t)
        print(f"[INFO] Best algorithm for \"{t}\" is {best_algo.algorithm},")
        print(f"       with average test loss {best_loss}, SD test loss {sd}")
        print(f"[INFO] Finding best model for this best algorithm...")
        X_train, y_train = df_t["discourse_text"], df_t["discourse_effectiveness"]
        df_test_t = df_test[df_test["discourse_type"] == t]
        X_test, y_test = df_test_t["discourse_text"], df_test_t["discourse_effectiveness"]
        best_models[t] = best_algo.get_best_model(X_train, y_train, export_results=True, discourse_type=t)
        loss = evaluate_model(best_models[t], X_test, y_test)
        print(f"[INFO] Best model for \"{t}\" is {best_models[t]}")
        print(f"       Test log loss: {loss}")
        final_results["Best algorithm"].append(model_name(best_models[t]))
        final_results["Test log loss"].append(round(loss, 3))

    df_final_results = pd.DataFrame(final_results)
    df_final_results.to_csv("best_algorithms.csv")

    return best_models, df_test


def model_name(model):
    mapping = {
        LogisticRegression: "Logistic regression",
        MultinomialNB: "Multinomial naive bayes",
        KNeighborsClassifier: "KNN",
        DecisionTreeClassifier: "Decision tree"
    }

    return mapping[type(model.get_params()["steps"][1][1])]


# Return the best Algorithm for the given dataset.
def choose_best_algorithm(data, discourse_type):
    algorithms = [
        HyperFinder(LogisticRegression(solver="newton-cg", random_state=3), LR_HYPERS),
        HyperFinder(MultinomialNB(), NB_HYPERS),
        HyperFinder(KNeighborsClassifier(), KNN_HYPERS),
        HyperFinder(DecisionTreeClassifier(random_state=4), DT_HYPERS),
    ]

    texts = data["discourse_text"]
    labels = data["discourse_effectiveness"]

    # Keep track of average and SD loss for each algorithm
    algo_losses = {
        "Algorithm": ["Logistic regression", "Multinomial naive Bayes", "KNN", "Decision tree"],
        "Average test log loss": [],
        "Standard deviation": [],
    }

    # Use stratified kfold if there is an imbalance in class distribution
    best_loss = None
    best_algo = None
    best_sd = None
    skf = StratifiedKFold(n_splits=OUTER_CV_K, shuffle=True, random_state=1)
    for algo in algorithms:
        print(f"[INFO] Trying algorithm {algo.algorithm}...")
        # Find algorithm with best average evaluation over the splits
        losses = []
        for train, test in skf.split(texts, labels):
            X_trainval, y_trainval = texts.iloc[train], labels.iloc[train]
            X_test, y_test = texts.iloc[test], labels.iloc[test]
            model = algo.get_best_model(X_trainval, y_trainval)
            model.fit(X_trainval, y_trainval)
            loss = evaluate_model(model, X_test, y_test)
            losses.append(loss)
        
        avg_loss = sum(losses) / OUTER_CV_K
        sd_loss = np.std(losses)
        algo_losses["Average test log loss"].append(round(avg_loss, 3))
        algo_losses["Standard deviation"].append(round(sd_loss, 3))
        if best_loss == None or avg_loss < best_loss:
            best_algo = algo
            best_loss = avg_loss
            best_sd = sd_loss

    # Save loss results for each algorithm
    losses_df = pd.DataFrame(algo_losses)
    losses_df.sort_values("Average test log loss")
    losses_df.to_csv(f"outer_cv_results/{discourse_type}.csv")

    return best_algo, best_loss, best_sd


def evaluate_model(model, X_test, y_test):
    return log_loss(y_test, model.predict_proba(X_test))


def predict(models, data):
    predictions = []
    for _, row in data.iterrows():
        kind = row["discourse_type"]
        text = row["discourse_text"]
        predictions.append(models[kind].predict_proba([text])[0])

    return np.array(predictions)


if __name__ == "__main__":
    initial = time.time()
    models, df_test = train_models()
    print("[INFO] The best models for each discourse type are:")
    for t, m in models.items():
        print(f"       {t}: {m}")
    
    predictions = predict(models, df_test)
    y_test = df_test["discourse_effectiveness"]
    print(f"[INFO] Overall test log loss: ", log_loss(y_test, predictions))
    final = time.time()
    print(f"[INFO] Total time taken: {final - initial} seconds")
