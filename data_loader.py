import numpy as np
import string
import pickle

# Loads data from filepath with num_cols into an array. Uses the ',' delimiter
# to distinguish between columns
# Returns a list of features and a list of labels.
def load_data(filepath):
    data = np.loadtxt(filepath, dtype=str, delimiter=',')
    features = data[:,:-1].astype(int)
    labels = data[:,-1]
    return features, labels


def save_tree(t, filepath):
    f = open(filepath, 'w+b')
    pickle.dump(t, f)
    f.close()


def load_tree(filepath):
    f = open(filepath, 'r+b')
    t = pickle.load(f)
    f.close()
    return t

# Returns two arrays. The first containing the max of each col
# and the second containing the min of each col.
def max_min_cols(array):
    max_array = np.zeros((len(array[0])), dtype=int)
    min_array = np.zeros((len(array[0])), dtype=int)

    # for each col:
    for i in range(len(array[0])):
        maximum = -1
        minimum = 9999999
        #for each row:
        for j in range(len(array)):
            cur_value = array[j][i]
            maximum = max(maximum, cur_value)
            minimum = min(minimum, cur_value)

        max_array[i] = maximum
        min_array[i] = minimum

    return max_array, min_array


'''
features, labels = load_data('data/train_sub.txt', 16)
max_array, min_array = max_min_cols(features)

features, labels = load_data('data/train_sub.txt', 16)
max_array, min_array = max_col_and_min_col_of_2d_array(features)

for j in range(0, len(max_array)):
    print(max_array[j])
print('')

for j in range(0, len(min_array)):
    print(min_array[j])

for j in range(0, len(min_array)):
    print(min_array[j])

    
number_of_each_label(labels)
'''


# Gets the number of occurences of each label in labels array
# Returns a dictionary with the count of each label.
def number_of_each_label(labels):
    dict = {c : 0 for c in list(string.ascii_uppercase)}
    for c in labels:
        dict[c] += 1
    print(dict)
    return dict
