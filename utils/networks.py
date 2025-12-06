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
        return config
    
    elif i==1: #square
        config = {
            'a': (['delta', 'alpha'], 4),
            'b': (['alpha', 'beta'], 4),
            'c': (['beta', 'gamma'], 4),
            'd': (['gamma', 'delta'], 4)
        }
        return config
    
    elif i==2: #square with one diagonal
        config = {
            'a': (['delta', 'alpha', 'ac_source'], 8),
            'b': (['alpha', 'beta'], 4),
            'c': (['beta', 'gamma', 'ac_source'], 8),
            'd': (['gamma', 'delta'], 4)
        }    
        return config
    
    elif i==3: #square with two diagonals
        config = {
            'a': (['delta', 'alpha', 'ac_source'], 8),
            'b': (['alpha', 'beta', 'bd_source'], 8),
            'c': (['beta', 'gamma', 'ac_source'], 8),
            'd': (['gamma', 'delta', 'bd_source'], 8)
        }    
        return config
    
    elif i==4:#pentagon
        config = {
            'a': (['epsilon', 'alpha'], 4),
            'b': (['alpha', 'beta'], 4),
            'c': (['beta', 'gamma'], 4),
            'd': (['gamma', 'delta'], 4),
            'e': (['delta', 'epsilon'], 4)
        }
        return config
    
    elif i==5: #binary triangle
        config = {
            'a': (['beta', 'gamma'], 2),
            'b': (['alpha', 'gamma'], 2),
            'c': (['alpha', 'beta'], 2)
        }
        return config
    
    elif i==6:#pentagon with 3 outputs
        config = {
            'a': (['epsilon', 'alpha'], 3),
            'b': (['alpha', 'beta'], 3),
            'c': (['beta', 'gamma'], 3),
            'd': (['gamma', 'delta'], 3),
            'e': (['delta', 'epsilon'], 3)
        }
        return config
    
    elif i==7: #square with two diagonals (tetrahedron)
        config = {
            'a': (['ad', 'ab', 'ac'], 8),
            'b': (['ab', 'bc', 'bd'], 8),
            'c': (['bc', 'cd', 'ac'], 8),
            'd': (['cd', 'ad', 'bd'], 8)
        }    
        return config
    
    else:
        raise(ValueError("Input should be in {0,1,2,3,4,5,6}"))