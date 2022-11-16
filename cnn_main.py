
import itertools
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

from cnn import CNN
from embedding import load_embeddings
from preprocess import clean_text, pad_sequence


torch.manual_seed(1)
torch.use_deterministic_algorithms(mode=True)


learn_rates = [0.0005, 0.001, 0.002]
batch_sizes = [32, 64]
max_feature_choices = [5000, 7500, 10000]
hyper_combos = list(itertools.product(learn_rates, batch_sizes, max_feature_choices))
EPOCHS = 40
EMBED_SIZE = 50


df = pd.read_csv("data/train.csv")
df.dropna()
df.drop(columns=["discourse_id", "essay_id"], inplace=True)
df["discourse_text"] = df["discourse_text"].apply(clean_text)
types = df["discourse_type"].unique()


def split_data(df, t):
    # Do a 60/20/20 train/val/test split on the given datast for discourse type t.
    df_t = df[df["discourse_type"] == t]
    df_t_trainval, df_t_test = train_test_split(
        df_t, test_size=0.2, random_state=8, stratify=df_t["discourse_effectiveness"]
    )
    df_t_train, df_t_val = train_test_split(
        df_t_trainval, test_size=0.25, random_state=3, stratify=df_t_trainval["discourse_effectiveness"]
    )

    X_train, y_train = df_t_train["discourse_text"], df_t_train["discourse_effectiveness"]
    X_test, y_test = df_t_test["discourse_text"], df_t_test["discourse_effectiveness"]
    X_val, y_val = df_t_val["discourse_text"], df_t_val["discourse_effectiveness"]

    return X_train, X_val, X_test, y_train, y_val, y_test


def to_sequences(X, tokenizer, maxlen):
    # Convert texts X to equal-length sequences using the 
    # tokenizer, with a maximum length for zero-padding.
    seqs = tokenizer.texts_to_sequences(X)
    seqs = [pad_sequence(seq, maxlen) for seq in seqs]
    return seqs


def get_dataloader(X, y, batch_size, shuffle):
    # convert effectiveness labels to numeric values
    encoder = LabelEncoder()
    labels = encoder.fit_transform(y)
    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(labels)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def multiclass_log_loss(pred, y):
    # Evaluation metric used by kaggle. This is a differentiable function.
    # Given y in [0, 1, 2] (class labels) and predicted probabilities p = [p0, p1, p2],
    # this formula was carefully defined as
    #
    # loss(y, p) = -(2-y)(1-y)/2 ln(p0) - (2-y)y ln(p1) + (1-y)y/2 ln(p2),
    #
    # so that 
    # loss(0, p) = -ln(p0),
    # loss(1, p) = -ln(p1),
    # loss(2, p) = -ln(p2).
    log_pred = torch.log(pred)

    def fy(n):
        return [int(-(2-n)*(1-n) / 2), int(-(2-n)*n), int((1-n)*n / 2)]

    y_forms = torch.tensor([fy(k) for k in y])
    return (log_pred * y_forms).sum() / len(pred)


def train_cnn(learn_rate, batch_size, max_features, train_loader, val_loader, embedding_matrix):
    # Train a CNN given the three hyperparamters on the train and validation loaders,
    # with the given embedding matrix. Return a dictionary with the training history,
    # including train/validation losses/accuracies per epoch, and also return the
    # model state corresponding to the model with lowest validation loss, along with
    # the actual minimum validation loss.
    model = CNN(embedding_matrix, 3, max_features, EMBED_SIZE)
    optimizer = Adam(model.parameters(), lr=learn_rate)
    loss_func = multiclass_log_loss

    # Keep track of losses and accuracy per epoch, as well as best model state
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = None
    best_model_state = None
    for i in range(EPOCHS):
        model.train()
        train_loss_sum, val_loss_sum, train_correct, val_correct = 0, 0, 0, 0

        # Predict each batch and backpropagate
        for x_batch, y_batch in train_loader:
            prediction = model(x_batch)
            loss = loss_func(prediction, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss * len(x_batch)
            train_correct += (prediction.argmax(1) == y_batch).type(torch.float).sum().item()
        
        # Evaluate model on validation set
        with torch.no_grad():
            model.eval()
            for x_batch, y_batch in val_loader:
                prediction = model(x_batch)
                val_loss_sum += loss_func(prediction, y_batch) * len(x_batch)
                val_correct += (prediction.argmax(1) == y_batch).type(torch.float).sum().item()

        train_loss_avg = train_loss_sum / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        val_loss_avg = val_loss_sum / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        if best_val_loss == None or val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_state = model.state_dict()

        history["train_acc"].append(float(train_acc))
        history["train_loss"].append(float(train_loss_avg))
        history["val_acc"].append(float(val_acc))
        history["val_loss"].append(float(val_loss_avg))

        print(f"      Epoch {i + 1}/{EPOCHS}")
        print(f"      Train loss: {train_loss_avg}, Train accuracy: {train_acc}")
        print(f"      Validation loss: {val_loss_avg}, Validation accuracy: {val_acc}\n")
    
    return history, best_model_state, min(history["val_loss"])


def evaluate_model(model_attrs, test_loader):
    # Evaluate the model with given attributes using against the test loader.
    max_features = model_attrs["hypers"][2]
    loss_func = multiclass_log_loss
    test_loss_sum = 0
    with torch.no_grad():
        model = CNN(model_attrs["embedding_matrix"], 3, max_features, EMBED_SIZE)
        model.load_state_dict(model_attrs["model_state"])
        model.eval()
        for x_batch, y_batch in test_loader:
            pred = model(x_batch)
            test_loss_sum += loss_func(pred, y_batch) * len(x_batch)

    test_loss = test_loss_sum / len(test_loader.dataset)
    print(f"      Test loss for type {t}: {test_loss}\n")
    return test_loss


best_models = dict()
test_sizes = dict()
start = time.time()
for t in types:
    # Set up a 60/20/20 train/validation/test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, t)

    # Find the 95th percentile of text lengths
    lengths = [len(txt.split(" ")) for txt in pd.concat((X_train, X_val, X_test))]
    len95 = lengths[-len(lengths)//20]

    # Try all combinations of hyperparameters, and find the one which
    # leads to a model with the best validation loss. Keep track of
    # various attributes of the best model found in this way.
    best_attributes = {
        "val_loss": None, "train_loss": None, "hypers": None, "model_state": None, 
        "history": None, "embedding_matrix": None, "test_loss": None
    }
    train_start = time.time()
    print(f"[CNN] Training network for type {t}...")
    for lr, batch_size, max_features in hyper_combos:
        print(f"      Trying hypers (learn rate, batch size, max features) = {lr, batch_size, max_features}...\n")
        # Tokenize all words in texts of type t
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(pd.concat((X_train, X_val, X_test)))

        # Convert texts to zero-padded sequences of indices.
        # Cap text length to the 95th percentile length.
        train_seqs, val_seqs = [to_sequences(X, tokenizer, len95) for X in [X_train, X_val]]

        # Convert each train/val/test sets into DataLoaders of sequences.
        train_loader = get_dataloader(train_seqs, y_train, batch_size, shuffle=True)
        val_loader = get_dataloader(val_seqs, y_val, batch_size, shuffle=False)

        # Load the embedding matrix
        embedding_matrix = load_embeddings(tokenizer.word_index, max_features, EMBED_SIZE)

        # Train a CNN with this combination of hyperparamters and get the training
        # history, and model state corresponding to lowest validation loss
        history, model_state, val_loss = train_cnn(
            lr, batch_size, max_features, train_loader, val_loader, embedding_matrix
        )

        print(f"      Best model with these hypers got validation loss {val_loss}\n")

        # Update best model, if the one found above is the best one so far
        if best_attributes["val_loss"] == None or val_loss < best_attributes["val_loss"]:
            best_attributes["val_loss"] = val_loss
            best_attributes["model_state"] = model_state
            best_attributes["hypers"] = (lr, batch_size, max_features)
            best_attributes["history"] = history
            best_attributes["embedding_matrix"] = embedding_matrix

    train_end = time.time()
    print(f"      Best hypers for '{t}' are {best_attributes['hypers']} with validation loss {best_attributes['val_loss']}")
    print(f"      Total time taken to train for type {t}: {train_end - train_start}s")

    # Evaluate the best model + best hyperparameter combo against test set
    # (Learning rate and batch size is not needed for evaluation - they are
    # only used to obtain the best model parameters)
    print("      Evaluating this best model with best hyperparameters...")
    best_max_features = best_attributes["hypers"][2]
    tokenizer = Tokenizer(num_words=best_max_features)
    tokenizer.fit_on_texts(pd.concat((X_train, X_val, X_test)))
    test_seqs = to_sequences(X_test, tokenizer, len95)
    test_loader = get_dataloader(test_seqs, y_test, 10, shuffle=False)
    best_attributes["test_loss"] = float(evaluate_model(best_attributes, test_loader))
    best_models[t] = best_attributes
    test_sizes[t] = len(test_loader.dataset)

end = time.time()

# Plot the training history of the best model found for each discourse type
for t in types:
    xrange = np.linspace(1, EPOCHS, EPOCHS)
    h = best_models[t]["history"]
    train_losses = [float(x) for x in h["train_loss"]]
    train_accs = [float(x) for x in h["train_acc"]]
    val_losses = [float(x) for x in h["val_loss"]]
    val_accs = [float(x) for x in h["val_acc"]]
    plt.plot(xrange, train_losses, color="orange")
    plt.plot(xrange, val_losses, color="red")
    plt.plot(xrange, train_accs, color="green")
    plt.plot(xrange, val_accs, color="blue")
    plt.xlabel("Epoch")
    plt.title(f"Training History for Finding Best Model for Discourse Type '{t}'")
    plt.legend(["Train loss", "Validation loss", "Train accuracy", "Validation accuracy"])
    plt.show()


# Export test losses and print overall test loss
overall_loss = 0
total_test_size = sum(list(test_sizes.values()))
overall_results = {
    "Discourse type": [], "Hyperparameters used": [],
    "Train loss": [], "Validation loss": [], "Test loss": []
}
for t in types:
    best_model = best_models[t]
    overall_results["Discourse type"].append(t)
    validation_loss = best_model["val_loss"]
    i = best_model["history"]["val_loss"].index(validation_loss)
    overall_results["Train loss"].append(round(best_model["history"]["train_loss"][i], 3))
    overall_results["Validation loss"].append(round(validation_loss, 3))
    test_loss = best_model["test_loss"]
    overall_results["Test loss"].append(round(test_loss, 3))
    hypers = ", ".join([f"{h}: {best_model['hypers'][i]}" for i, h in enumerate(["learn rate", "batch size", "max features"])])
    overall_results["Hyperparameters used"].append(hypers)
    overall_loss += test_loss * test_sizes[t] / total_test_size

overall_results_df = pd.DataFrame(overall_results)
overall_results_df.to_csv("cnn_final_models.csv")

print(f"Overall loss: {overall_loss}")
print(f"Total time taken: {end - start}s")
