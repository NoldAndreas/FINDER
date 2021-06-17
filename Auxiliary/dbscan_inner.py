# Fast inner loop for DBSCAN.
# Author: Lars Buitinck
# License: 3-clause BSD
#
# cython: boundscheck=False, wraparound=False
# https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/cluster/_dbscan_inner.pyx

#cimport cython
#from libcpp.vector cimport vector
#cimport numpy as np
import numpy as np

#np.import_array()


# Work around Cython bug: C++ exceptions are not caught unless thrown within
# a cdef function with an "except +" declaration.
#cdef inline void push(vector[np.npy_intp] &stack, np.npy_intp i) except +:
#    stack.push_back(i)


def dbscan_inner(is_core,neighborhoods,labels):
    #cdef np.npy_intp i, label_num = 0, v
    #cdef np.ndarray[np.npy_intp, ndim=1] neighb
    #cdef vector[np.npy_intp] stack
    label_num = 0
    stack = [];

    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:
            continue

        # Depth-first search starting from i, ending at the non-core points.
        # This is very similar to the classic algorithm for computing connected
        # components, the difference being that we label non-core points as
        # part of a cluster (component), but don't expand their neighborhoods.
        while True:
            if labels[i] == -1:
                labels[i] = label_num
                if is_core[i]:
                    neighb = neighborhoods[i]
                    for i in range(neighb.shape[0]):
                        v = neighb[i]
                        if labels[v] == -1:
                            stack.append(v)
                            #push(stack, v)

            if len(stack) == 0:
                break
            i = stack[-1]
            stack = stack[:-1];

        label_num += 1
