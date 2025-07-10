# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:25:58 2024

@author: Girardin
"""
import numpy as np


def angled_phip(t):
    state = np.array([np.cos(t), 0, 0, np.sin(t)])
    return state

def angled_psip(t):
    state = np.array([0, np.cos(t), np.sin(t), 0])
    return state

def angled_phim(t):
    state = np.array([np.cos(t), 0, 0, -np.sin(t)])
    return state

def angled_psim(t):
    state = np.array([0, np.cos(t), -np.sin(t), 0])
    return state

def get_state(i, v=1):
    """return the state for:
        i=0:cos(t)|00> + sin(t)|11> for t=pi/4
        i=1:cos(t)|00> + sin(t)|11> for t=pi/8
        i=2:cos(t)|00> + sin(t)|11> for t=pi/16
        i=3:cos(t)|01> + sin(t)|10> for t=pi/4
        i=4:cos(t)|01> + sin(t)|10> for t=pi/8
        i=5:cos(t)|01> + sin(t)|10> for t=pi/16
        """
        
    if i==0:
        state = angled_phip(np.pi/4)
    elif i==1:
        state = angled_phip(np.pi/8)
    elif i==2:
        state = angled_phip(np.pi/16)
    elif i==3:
        state = angled_psip(np.pi/4)
    elif i==4:
        state = angled_psip(np.pi/8)  
    elif i==5:
        state = angled_psip(np.pi/16)
    elif i==6:
        state = angled_phim(np.pi/4)
    elif i==7:
        state = angled_psim(np.pi/4)
    else:
        raise(ValueError("input should be in {0, ..., 7}"))
    
    state = np.reshape(state, [1,4])
    out = state*np.conj(state.T)
    noisy_state = v*out + (1-v)*np.eye(4)/4
    if v>1 or v<0:
        print("Warning: value of v is not in [0,1]")
    return (noisy_state, [2,2])