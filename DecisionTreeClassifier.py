import numpy as np
import pandas as pd
import scipy.stats as st
from time import time
from collections import deque

class Node:
    def __init__(self, feature=-1, split=None, impurity=np.inf):
        # Split feature
        self.feature = feature
        # Split criterion
        self.split = split
        self.impurity = impurity
        self.children = []
        self.leaf = False
        self.label = None
        self.depth = 0
        
    def get_split_indices(self, X, intersect_with=None):
        X_feature = X[:, self.feature]

        # Categorical feature
        if isinstance(self.split, np.ndarray):
            splitted_data = []
            for value in self.split:
                indices = np.asarray(X_feature == value).nonzero()[0]
                if intersect_with is not None:
                    indices = np.intersect1d(indices, intersect_with, assume_unique=True)
                splitted_data.append(indices)
                    
            return splitted_data
        
        # Numerical feature
        indices_left = np.asarray(X_feature < self.split).nonzero()[0]
        indices_right = np.asarray(X_feature >= self.split).nonzero()[0]

        if intersect_with is not None:
            indices_left = np.intersect1d(indices_left, intersect_with, assume_unique=True)
            indices_right = np.intersect1d(indices_right, intersect_with, assume_unique=True)

        return [indices_left, indices_right]

    def __get_probs(self, y):
        unique_y = np.unique(y)
        probs = np.zeros(len(unique_y))
        y_len = len(y)
        for i, y_i in enumerate(unique_y):
            probs[i] = len(y[y == y_i]) / y_len
        
        return probs

    def calc_impurity(self, y, criterion, store=False):
        probs = self.__get_probs(y)
        impurity = None
        if criterion == 'entropy':
            impurity = self.__calc_entropy(probs)
        elif criterion == 'gini':
            impurity = self.__calc_gini(probs)
        if store:
                self.impurity = impurity
        return impurity
            
    def impurity_for_split(self, X, y, criterion):
        splitted_indices = self.get_split_indices(X)  
        impurities = np.zeros(len(splitted_indices))
        len_X = len(X)
        for index, branch_indices in enumerate(splitted_indices):
            y_branch = y[branch_indices]
            impurities[index] = self.calc_impurity(y_branch, criterion) * len(y_branch)
        return np.sum(impurities) / len_X
    
    def __calc_entropy(self, probs):
        entropy = -np.sum(probs * np.log(probs + 10e-8))
        return entropy
    
    def __calc_gini(self, probs):
        gini = 1 - np.sum(probs ** 2)
        return gini

    def __str__(self):
        return 'Node(leaf={})'.format(self.leaf)
    
    def __repr__(self):
        return 'Node(leaf={})'.format(self.leaf)


class DecisionTreeClassifier:
    def __init__(self, tol=0.5, max_depth=10, min_members=50, criterion='entropy'):
        self.tol = tol
        self.tree_depth = 0
        self.max_depth = max_depth
        self.min_members = min_members
        self.criterion = criterion
    
    def fit(self, X, y):
        self.tree_ = Node()
        self.features_ = X.columns if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
        self.important_features_ = []
        X_ = self.__get_values(X)
        y_ = self.__get_values(y)
        feature_types = [self.__check_type(X_[:, column]) for column in range(X.shape[1])]
        self.__generate_tree(self.tree_, X_, y_, feature_types)
    
    def __generate_tree(self, tree, X, y, feature_types):
        if len(y) <= self.min_members or tree.calc_impurity(y, store=True, criterion=self.criterion) < self.tol:
            self.__label_node(tree, y)
            return
        
        best_feature_split = self.__split_attribute(tree, X, y, feature_types)        
        tree.feature = best_feature_split[0]
        tree.split = best_feature_split[1]
        
        if tree.feature is None or tree.split is None:
            self.__label_node(tree, y)
            return
        
        self.important_features_.append(self.features_[tree.feature])

        splitted_data = tree.get_split_indices(X)
        num_branches = len(splitted_data)
        if num_branches < 2:
            self.__label_node(tree, y)
            return
        elif num_branches == 2:
            if len(splitted_data[0]) == 0 or len(splitted_data[1]) == 0:
                self.__label_node(tree, y)
                return
                
        for branch_indices in splitted_data:
            new_node = Node()
            tree.children.append(new_node)
            self.__generate_tree(new_node, X[branch_indices], y[branch_indices], feature_types)
        
    
    def __split_attribute(self, tree, X, y, feature_types):
        min_entropy = np.inf
        entropy = min_entropy
        best_feature = None
        best_split_value = None
        for feature in range(X.shape[1]):
            tree.feature = feature
            X_feature = X[:, feature]
            dtype = feature_types[feature]
            if dtype == 'cat':
                tree.split = np.unique(X_feature)
                if len(tree.split) < 2:
                    continue
                entropy = tree.impurity_for_split(X, y, criterion=self.criterion)
                if entropy < min_entropy:
                    min_entropy = entropy
                    best_feature = feature
                    best_split_value = tree.split
            else:
                X_feature_sorted_indices = np.argsort(X_feature)
                X_feature_sorted = X_feature[X_feature_sorted_indices]
                y_sorted = y[X_feature_sorted_indices]
                thresholds = (X_feature_sorted[1:] + X_feature_sorted[:-1])/2
                thresholds_len = len(thresholds)
                for value_index, value in enumerate(thresholds):
                    if (value_index < thresholds_len - 1) and (y_sorted[value_index] == y_sorted[value_index+1] or thresholds[value_index] == thresholds[value_index+1]):
                        continue
                    
                    tree.split = value
                    entropy = tree.impurity_for_split(X, y, criterion=self.criterion)

                    if entropy < min_entropy:
                        min_entropy = entropy
                        best_feature = feature
                        best_split_value = tree.split
                
        return best_feature, best_split_value
    
    def __label_node(self, node, y):
        most_frequent = st.mode(y)[0]
        rand = np.random.randint(len(most_frequent))
        node.leaf = True
        node.label = most_frequent[rand]

    def __check_type(self, data):
        try:
            number_data = data.astype(np.number)
            if np.all(np.mod(number_data, 1) == 0):
                return 'cat' if len(np.unique(data)) / len(data) <= 0.05 else 'num'
            return 'num'
        except ValueError:
            return 'cat'
        
    def __get_values(self, data):
        if isinstance(data, np.ndarray):
            return data
        
        return data.values

    def predict(self, X):
        X_ = self.__get_values(X)
        pred = np.full(X_.shape[0], -1)
        self.__decide(self.tree_, X_, pred, np.arange(X_.shape[0]))
        return pred

    def __decide(self, node, X, pred, indices):
        if node.leaf:
            pred[indices] = node.label
            return
            
        branches = node.get_split_indices(X, indices)
        for index, branch in enumerate(branches):
            self.__decide(node.children[index], X, pred, branch)

    def better_predict(self, X):
        pred = np.full(X.shape[0], -1)
        stack = deque([self.tree_])
        indices_stack = deque([np.arange(X.shape[0])])
        while len(stack) > 0:
            current = stack.pop()
            current_indices = indices_stack.pop()
            while current:
                len_children = len(current.children)
                if len_children > 0:
                    tree_split = current.get_split_indices(X, intersect_with=current_indices)
                    for index in range(len_children - 1, 0, -1):
                        stack.append(current.children[index])
                        indices_stack.append(tree_split[index])

                    current = current.children[0]
                    current_indices = tree_split[0]
                else:
                    # leaf
                    pred[current_indices] = current.label
                    current = None


        return pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return y_pred[y == y_pred].size / y_pred.size
    
    def better_score(self, X, y):
        y_pred = self.better_predict(X)
        return y_pred[y == y_pred].size / y_pred.size
        