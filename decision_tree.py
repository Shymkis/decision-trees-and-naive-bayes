import numpy as np

class Decision_Tree:
    def __init__(self, node_label, branches=None):
        self.node_label = node_label
        self.branches = branches if branches is not None else []

    def add_branch(self, branch):
        self.branches.append(branch)

    def decide(self, example):
        for b in self.branches:
            if b["branch_label"] == example[self.node_label]:
                return b["node"].decide(example)
            if isinstance(b["branch_label"], str):
                if "<=" in b["branch_label"] and example[self.node_label] <= float(b["branch_label"][2:]):
                    return b["node"].decide(example)
                if ">" in b["branch_label"] and example[self.node_label] > float(b["branch_label"][1:]):
                    return b["node"].decide(example)
        return self.node_label

    def __str__(self, level=1):
        ret = str(self.node_label) + "\n"
        for b in self.branches:
            ret += "\t"*level + str(b["branch_label"]) + ": " + b["node"].__str__(level + 1)
        return ret


def all_equal(examples):
    vals = examples["label"].to_numpy()
    return (vals[0] == vals).all()

def entropy(probs):
    return -(probs*np.log2(probs)).sum()

def gini(probs):
    return (probs*(1 - probs)).sum()

def impurity(examples, measure):
    totals = examples["label"].value_counts()
    probs = totals/len(examples.index)
    return measure(probs)

def importance(feat, examples, measure):
    # Information Gain
    before = impurity(examples, measure)
    after = float("inf")
    split = None
    if isinstance(feat["values"], tuple):
        # Handle numeric data
        exs = examples.sort_values(by=[feat["feature"]])
        mids = set()
        for i in range(len(exs.index) - 1):
            e1 = exs.iloc[i]
            e2 = exs.iloc[i + 1]
            lab1 = e1["label"]
            lab2 = e2["label"]
            if lab1 != lab2:
                mids.add((e1[feat["feature"]] + e2[feat["feature"]])/2)
        for s in mids:
            low_exs = exs[exs[feat["feature"]] <= s]
            high_exs = exs[exs[feat["feature"]] > s]
            aft = (len(low_exs)/len(examples))*impurity(low_exs, measure) + \
                (len(high_exs)/len(examples))*impurity(high_exs, measure)
            if aft < after:
                after = aft
                split = s
    else:
        after = 0
        for v in feat["values"]:
            exs = examples[examples[feat["feature"]] == v]
            after += (len(exs.index)/len(examples.index))*impurity(exs, measure)
    return before - after, split

def argmax(features, examples, impurity_measure):
    F = None
    V = float("-inf")
    split = None
    for i, f in features.iterrows():
        v, s = importance(f, examples, impurity_measure)
        if v > V:
            F, V, split = f, v, s
    return F, split

def train_model(examples, features, parent_examples=None, impurity_measure=entropy, max_depth=float("inf"), depth=0):
    if examples.empty:
        return Decision_Tree(parent_examples["label"].mode().iloc[0])
    elif all_equal(examples):
        return Decision_Tree(examples["label"].iloc[0])
    elif features.empty or depth == max_depth:
        return Decision_Tree(examples["label"].mode().iloc[0])
    else:
        F, split = argmax(features, examples, impurity_measure)
        tree = Decision_Tree(F["feature"])
        if split:
            # Handle numeric data
            split = round(split, 9)
            low_exs = examples[examples[F["feature"]] <= split]
            low_subtree = train_model(low_exs, features, examples, impurity_measure, max_depth, depth + 1)
            tree.add_branch({"branch_label": "<=" + str(split), "node": low_subtree})

            high_exs = examples[examples[F["feature"]] > split]
            high_subtree = train_model(high_exs, features, examples, impurity_measure, max_depth, depth + 1)
            tree.add_branch({"branch_label": ">" + str(split), "node": high_subtree})
        else:
            for v in F["values"]:
                feats = features[features["feature"] != F["feature"]]
                exs = examples[examples[F["feature"]] == v]
                subtree = train_model(exs, feats, examples, impurity_measure, max_depth, depth + 1)
                tree.add_branch({"branch_label": v, "node": subtree})
        return tree
