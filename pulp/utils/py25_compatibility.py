#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

"""
    The pulp.utils.py25_compatibility contains a few items that are new to
    Python 2.6 in order to mantain compatibility with Python 2.5:
    some less fortunate collaborators are stuck with it.
"""



# this function is part of the standard library as of Python 2.6
def _py25_combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

class _py25_ABCMeta(type):
    pass

def _py25_abstractmethod(func):
    return func

