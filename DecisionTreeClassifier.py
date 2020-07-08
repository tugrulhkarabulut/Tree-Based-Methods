import numpy as np
import pandas as pd
from time import time
from sklearn import tree

class Node:
    def __init__(self, feature=-1, split=None, entropy=0):
        # Split feature
        self.feature = feature
        # Split criterion
        self.split = split
        self.entropy = entropy
        self.children = []
        self.parent = None
        self.leaf = False
        self.label = None
        self.depth = 0
        
    def get_elements(self, X, y=None):
        # Categorical feature
        if isinstance(self.split, list):
            splitted_data = []
            for value in self.split:
                indices = X.iloc[:, self.feature] == value
                if y is None:
                    splitted_data.append(X[indices])
                else:
                    splitted_data.append((X[indices], y[indices]))
                    
            return splitted_data
        
        # Numerical feature
        indices_left = X.iloc[:, self.feature] < self.split
        indices_right = X.iloc[:, self.feature] >= self.split
        
        if y is None:
            return [X[indices_left], X[indices_right]]
        
        return [(X[indices_left], y[indices_left]), (X[indices_right], y[indices_right])]
            
    def entropy_for_split(self, X, y):
        splitted_data = self.get_elements(X, y)
        entropies = np.zeros(len(splitted_data))
        for index, data in enumerate(splitted_data):
            entropies[index] = self.calc_entropy(data[1]) * len(data[0])
        return np.sum(entropies) / len(X)
    
    def calc_entropy(self, y, store=False):
        unique_y = np.unique(y)
        probs = np.zeros(len(unique_y))
        y_len = len(y)
        for i, y_i in enumerate(unique_y):
            probs[i] = len(y[y == y_i]) / y_len
        
        entropy = -np.sum(probs * np.log2(probs + 10e-8))
        if store:
            self.entropy = entropy
        return entropy


class DecisionTreeClassifier:
    def __init__(self, tol=0.5, max_depth=10, min_members=50):
        self.tol = tol
        self.tree = None
        self.tree_depth = 0
        self.max_depth = max_depth
        self.min_members = min_members
    
    def fit(self, X, y):
        self.tree_ = Node()
        self.__generate_tree(self.tree_, X, y)
    
    def __generate_tree(self, tree, X, y):
        if len(y) <= self.min_members or tree.calc_entropy(y, store=True) < self.tol:
            self.__label_node(tree, y)
            return
        
        best_feature_split = self.__split_attribute(tree, X, y)        
        tree.feature = best_feature_split[0]
        tree.split = best_feature_split[1]
        
        if tree.feature is None or tree.split is None:
            self.__label_node(tree, y)
            return
        
        splitted_data = tree.get_elements(X, y)
        
        if len(splitted_data) < 2:
            self.__label_node(tree, y)
            return
                
        for el in splitted_data:
            new_node = Node()
            tree.children.append(new_node)
            new_node.parent = tree
            self.__generate_tree(new_node, el[0], el[1])
        
    
    def __split_attribute(self, tree, X, y):
        min_entropy = 100 if tree.parent is None else tree.parent.entropy
        entropy = min_entropy
        best_feature = None
        best_split_value = None
        for index, feature in enumerate(X.columns):
            tree.feature = index
            if X[feature].dtype.name == 'category':
                tree.split = list(X[feature].unique())
                if len(tree.split) < 2:
                    continue
                entropy = tree.entropy_for_split(X, y)
                if entropy < min_entropy:
                    min_entropy = entropy
                    best_feature = index
                    best_split_value = tree.split
            else:
                X_feature_sorted = X.iloc[1:, index].sort_values()
                y_sorted = y[X_feature_sorted.index].values
                X_feature_sorted_values = X_feature_sorted.values
                thresholds = (X_feature_sorted_values[1:] + X_feature_sorted_values[:-1])/2
                thresholds_len = len(thresholds)
                for value_index, value in enumerate(thresholds):
                    if (value_index < thresholds_len - 1) and (y_sorted[value_index] == y_sorted[value_index+1] or thresholds[value_index] == thresholds[value_index+1]):
                        continue
                    
                    tree.split = value
                    entropy = tree.entropy_for_split(X, y)

                    if entropy < min_entropy:
                        min_entropy = entropy
                        best_feature = index
                        best_split_value = tree.split
                
        return best_feature, best_split_value
    
    def __label_node(self, node, y):
        most_frequent = y.mode()
        rand = np.random.randint(len(most_frequent))
        node.leaf = True
        node.label = y.mode()[rand]
    
    def predict(self, X):
        pred = pd.Series(-1, X.index)
        self.__decide(self.tree_, X, pred)
        return pred
    def __decide(self, node, X, pred):
        if node.leaf:
            pred[X.index] = node.label
            return
            
        branches = node.get_elements(X)
        for index, branch in enumerate(branches):
            self.__decide(node.children[index], branch, pred)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return y_pred[y == y_pred].size / y_pred.size
        