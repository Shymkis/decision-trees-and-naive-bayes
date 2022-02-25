import numpy as np
import pandas as pd

def pixel_counts(p_vecs):
    pixels = [sum(p_vec) for p_vec in zip(*p_vecs)]
    return {i: v for i, v in enumerate(pixels)}

def train_model(train_df, k):
    # Label probabilities
    label_probs = train_df.label.value_counts()/len(train_df.index)
    # Feature counts
    f_given_label_counts = pd.DataFrame({
        label: pixel_counts(train_df.loc[train_df.label == label, "f_vec"]) for label in label_probs.keys()
    }).fillna(0)
    # Laplace smoothing
    f_given_label_counts += k
    # Feature probabilities
    f_given_label_probs = f_given_label_counts.apply(lambda x : x/(train_df.label.value_counts()[x.name] + 2*k))
    return label_probs, f_given_label_probs

def predict(nb_model, f_vec):
    label_probs, f_given_label_probs = nb_model
    label_values = {}
    for label in label_probs.keys():
        # Slowest segment of all code
        probs = [f_given_label_probs.loc[i, label] if f else 1 - f_given_label_probs.loc[i, label] for i, f in enumerate(f_vec)]
        probs.append(label_probs[label])
        label_values[label] = sum(np.log(probs))
    return max(label_values, key=label_values.get)
