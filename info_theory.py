from dataclasses import dataclass
from math import log2
import multiprocessing

def bind_float(i: float, threshold: float):
    if i > threshold:
        return 1.0
    return 0.0

def set_or_add_dict(dic, key, value):
    if key in dic:
        dic[key] += value
        return False
    dic[key] = value
    return True

def compute_cond_mut(name, bound_values, test_data):
    cond = []
    mutual = []
    ep_num = 0
    for epoch_hidden in bound_values:
        class_dicts = [dict() for i in range(5)]
        general_dict = dict()
        for tup in epoch_hidden:
            set_or_add_dict(class_dicts[tup[0]], tup[1], 1)
            set_or_add_dict(general_dict, tup[1], 1)

        tot = 0
        for d in class_dicts:
            probs = [i / len(test_data) * 5 for i in d.values()]
            tot += 0.2 * -1.0 * sum([i * log2(i) for i in probs])
        cond.append(tot)

        mut_probs = [i / len(test_data) for i in general_dict.values()]
        h = -1.0 * sum(i * log2(i) for i in mut_probs)

        mutual.append(h - tot)
        print(name,"  ", ep_num, "   ", tot, "  ", h - tot)
        ep_num+=1
    return cond, mutual