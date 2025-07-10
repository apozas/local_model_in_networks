# -*- coding: utf-8 -*-
"""
definition of different networks
"""

def get_network(i):
    """return the config of the network for:
        i=0:triangle
        i=1:square
        i=2:square with one diagonal
        i=3:square with two diagonals
        i=4:pentagon
        """
        
    if i==0: #triangle
        config = {
            'a': (['beta', 'gamma'], 4),
            'b': (['alpha', 'gamma'], 4),
            'c': (['alpha', 'beta'], 4)
        }
        dict_sources = {
            'alpha': ['b', 'c'],
            'beta': ['a', 'c'],
            'gamma': ['a', 'b'],
        }
        dict_party = {
            'a': ['beta', 'gamma'],
            'b': ['gamma', 'alpha'],
            'c': ['alpha', 'beta'],
        }
        return config, dict_sources, dict_party
    elif i==1: #square
        config = {
            'a': (['delta', 'alpha'], 4),
            'b': (['alpha', 'beta'], 4),
            'c': (['beta', 'gamma'], 4),
            'd': (['gamma', 'delta'], 4)
        }
        dict_sources = {
            'alpha': ['a', 'b'],
            'beta': ['b', 'c'],
            'gamma': ['c', 'd'],
            'delta': ['d', 'a']
        }
        dict_party = {
            'a': ['delta', 'alpha'],
            'b': ['alpha', 'beta'],
            'c': ['beta', 'gamma'],
            'd': ['gamma', 'delta']
        }
        return config, dict_sources, dict_party
    elif i==2: #square with one diagonal
        config = {
            'a': (['delta', 'alpha', 'ac_source'], 8),
            'b': (['alpha', 'beta'], 4),
            'c': (['beta', 'gamma', 'ac_source'], 8),
            'd': (['gamma', 'delta'], 4)
        }    
        dict_sources = {
            'alpha': ['a', 'b'],
            'beta': ['b', 'c'],
            'gamma': ['c', 'd'],
            'delta': ['d', 'a'],
            'ac_source': ['a', 'c'],
        }
        dict_party = {
            'a': ['delta', 'alpha', 'ac_source'],
            'b': ['alpha', 'beta'],
            'c': ['beta', 'gamma', 'ac_source'],
            'd': ['gamma', 'delta']
        }
        return config, dict_sources, dict_party
    elif i==3: #square with two diagonals
        config = {
            'a': (['delta', 'alpha', 'ac_source'], 8),
            'b': (['alpha', 'beta', 'bd_source'], 8),
            'c': (['beta', 'gamma', 'ac_source'], 8),
            'd': (['gamma', 'delta', 'bd_source'], 8)
        }    
        dict_sources = {
            'alpha': ['a', 'b'],
            'beta': ['b', 'c'],
            'gamma': ['c', 'd'],
            'delta': ['d', 'a'],
            'ac_source': ['a', 'c'],
            'bd_source': ['b', 'd']
        }
        dict_party = {
            'a': ['delta', 'alpha', 'ac_source'],
            'b': ['alpha', 'beta', 'bd_source'],
            'c': ['beta', 'gamma', 'ac_source'],
            'd': ['gamma', 'delta', 'bd_source']
        }
        return config, dict_sources, dict_party
    elif i==4:#pentagon
        config = {
            'a': (['epsilon', 'alpha'], 4),
            'b': (['alpha', 'beta'], 4),
            'c': (['beta', 'gamma'], 4),
            'd': (['gamma', 'delta'], 4),
            'e': (['delta', 'epsilon'], 4)
        }
        dict_sources = {
            'alpha': ['a', 'b'],
            'beta': ['b', 'c'],
            'gamma': ['c', 'd'],
            'delta': ['d', 'e'],
            'epsilon': ['e', 'a']
        }
        dict_party = {
            'a': ['epsilon', 'alpha'],
            'b': ['alpha', 'beta'],
            'c': ['beta', 'gamma'],
            'd': ['gamma', 'delta'],
            'e': ['delta', 'epsilon']
        }
        return config, dict_sources, dict_party
    elif i==5: #binary triangle
        config = {
            'a': (['beta', 'gamma'], 2),
            'b': (['alpha', 'gamma'], 2),
            'c': (['alpha', 'beta'], 2)
        }
        dict_sources = {
            'alpha': ['b', 'c'],
            'beta': ['a', 'c'],
            'gamma': ['a', 'b'],
        }
        dict_party = {
            'a': ['beta', 'gamma'],
            'b': ['gamma', 'alpha'],
            'c': ['alpha', 'beta'],
        }
        return config, dict_sources, dict_party
    
    elif i==6:#pentagon with 3 outputs
        config = {
            'a': (['epsilon', 'alpha'], 3),
            'b': (['alpha', 'beta'], 3),
            'c': (['beta', 'gamma'], 3),
            'd': (['gamma', 'delta'], 3),
            'e': (['delta', 'epsilon'], 3)
        }
        dict_sources = {
            'alpha': ['a', 'b'],
            'beta': ['b', 'c'],
            'gamma': ['c', 'd'],
            'delta': ['d', 'e'],
            'epsilon': ['e', 'a']
        }
        dict_party = {
            'a': ['epsilon', 'alpha'],
            'b': ['alpha', 'beta'],
            'c': ['beta', 'gamma'],
            'd': ['gamma', 'delta'],
            'e': ['delta', 'epsilon']
        }
        return config, dict_sources, dict_party
    
    else:
        raise(ValueError("Input should be in {0,1,2,3,4,5,6}"))