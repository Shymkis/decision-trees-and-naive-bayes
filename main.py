import decision_tree
import naive_bayes
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

def extract_features(examples, keep_cols=None, drop_cols=None, cont_cols=None):
    if keep_cols:
        features = pd.DataFrame({"feature": examples[keep_cols].columns})
    elif drop_cols:
        features = pd.DataFrame({"feature": examples.drop(columns=drop_cols).columns})
    values = []
    for col in features["feature"]:
        if cont_cols and col in cont_cols:
            values.append((examples[col].min(), examples[col].max()))
        else:
            values.append(examples[col].unique())
    features["values"] = values
    return features

def n_folds(N, df):
    folds = []
    for x in range(N):
        train_data = []
        test_data = []
        for i, row in df.iterrows():
            if i % N == x:
                test_data.append(row)
            else:
                train_data.append(row)
        folds.append({"train": pd.DataFrame(train_data), "test": pd.DataFrame(test_data)})
    return folds

def accuracy(df):
    correct = sum(df.apply(lambda row : int(row.prediction == row.label), axis=1))
    return correct/len(df.index)*100

if __name__ == "__main__":
    # examples = pd.read_csv("datasets/letter-recognition.csv", encoding="latin-1").rename(columns={"letter": "label"})
    # examples["f_vec"] = examples.drop(columns=["label"]).apply(lambda x : x//8).values.tolist()
    # features = extract_features(examples, drop_cols=["label", "f_vec"], cont_cols=examples.columns.tolist())

    examples = pd.read_csv("datasets/zoo.csv", encoding="latin-1").rename(columns={"class": "label"})
    examples["f_vec"] = examples.drop(columns=["label", "name", "legs"]).values.tolist()
    features = extract_features(examples, drop_cols=["label", "f_vec", "name"])

    # Shuffle data
    np.random.seed(100)
    examples = examples.sample(frac=1).reset_index(drop=True)

    # N-fold cross-validation
    N = 5
    folds = n_folds(N, examples)

    train_accuracy = test_accuracy = 0
    for fold in tqdm(folds):
        train_df = fold["train"]
        test_df = fold["test"]

        # Train model
        tree = decision_tree.train_model(train_df, features, max_depth=5)

        # Make predictions
        train_df["prediction"] = train_df.apply(lambda row : tree.decide(row), axis=1)
        test_df["prediction"] = test_df.apply(lambda row : tree.decide(row), axis=1)

        # Calculate accuracies
        train_accuracy += accuracy(train_df)
        test_accuracy += accuracy(test_df)
    print("Average DT training set accuracy: " + str(round(train_accuracy / N, 2)) + "%")
    print("Average DT testing set accuracy: " + str(round(test_accuracy / N, 2)) + "%")
    print()

    train_accuracy = test_accuracy = 0
    for fold in tqdm(folds):
        train_df = fold["train"]
        test_df = fold["test"]

        # Train model
        nb_model = naive_bayes.train_model(train_df, k=1)

        # Make predictions
        train_df["prediction"] = train_df.apply(lambda row : naive_bayes.predict(nb_model, row.f_vec), axis=1)
        test_df["prediction"] = test_df.apply(lambda row : naive_bayes.predict(nb_model, row.f_vec), axis=1)

        # Calculate accuracies
        train_accuracy += accuracy(train_df)
        test_accuracy += accuracy(test_df)
    print("Average NB training set accuracy: " + str(round(train_accuracy / N, 2)) + "%")
    print("Average NB testing set accuracy: " + str(round(test_accuracy / N, 2)) + "%")
    print()
