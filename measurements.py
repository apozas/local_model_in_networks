# -*- coding: utf-8 -*-
"""
definition of some useful measurements
"""

import numpy as np
from itertools import product

def hconj(A):
    return np.transpose(np.conj(A))

## 2 partite measurements
def elegant_family(t):
    oosr3=1/np.sqrt(3)
    eta_j=[oosr3, -oosr3, -oosr3, oosr3]
    pi=np.pi
    phi_j=[pi/4, 7*pi/4, 3*pi/4, 5*pi/4]
    def m_j(j):
        '''j in {1,2,3,4}'''
        return np.array([np.sqrt((1-eta_j[j-1])/2)* np.exp(1j*phi_j[j-1]/2), 
                          np.sqrt((1+eta_j[j-1])/2)* np.exp(-1j*phi_j[j-1]/2)])
    def neg_m_j(j):
        return np.array([np.sqrt((1+eta_j[j-1])/2)* np.exp(1j*(phi_j[j-1]+np.pi)/2), 
                          np.sqrt((1-eta_j[j-1])/2)* np.exp(-1j*(phi_j[j-1]+np.pi)/2)])
        
    def Phi_j(j):
        return (np.sqrt(3)+ np.exp(1j*t))/(2*np.sqrt(2))*np.kron(m_j(j), neg_m_j(j)) + (np.sqrt(3)- np.exp(1j*t))/(2*np.sqrt(2))*np.kron(neg_m_j(j), m_j(j))

    
    m0=np.reshape(Phi_j(1), [1,4])
    m1=np.reshape(Phi_j(2), [1,4])
    m2=np.reshape(Phi_j(3), [1,4])
    m3=np.reshape(Phi_j(4), [1,4])
    # print(m0)
    
    M0 = m0*hconj(m0)
    M1 = m1*hconj(m1)
    M2 = m2*hconj(m2)
    M3 = m3*hconj(m3)
    
    POVM = [[M0, M1, M2, M3]]
    return POVM
    
def TCfamily(u2):
    u=0
    # u=.5
    # u2=np.sqrt(0.85)
    # u2=np.sqrt(0.63)
    m2=np.array([[u,0,0,np.sqrt(1-u**2)]])
    m3=np.array([[np.sqrt(1-u**2),0,0,-u]])
    m0=np.array([[0,u2,np.sqrt(1-u2**2),0]])
    m1=np.array([[0,np.sqrt(1-u2**2),-u2,0]])
    
    M0 = m0*hconj(m0)
    M1 = m1*hconj(m1)
    M2 = m2*hconj(m2)
    M3 = m3*hconj(m3)
    POVM = [[M0, M1, M2, M3]]
    return POVM

## 3 partite measurements (8 outcomes)

def GHZ_basis_measurementv1(): #discussed with Alex (doesnt sum to Id)
    M=[]
    GHZ = np.zeros([8,8])
    GHZ[0][0]=1/2
    GHZ[7][7]=1/2
    GHZ[0][7]=1/2
    GHZ[7][0]=1/2
    Z=np.array([[1, 0],[0,-1]])
    X=np.array([[0,1],[1,0]])
    for a0, a1, a2 in product([0,1], repeat=3):
        Mi=np.dot(
            np.kron(
            np.kron(np.linalg.matrix_power(Z, a0), 
                    np.linalg.matrix_power(X, a1)), 
            np.linalg.matrix_power(X, a2)), 
            GHZ)
        M.append(Mi)
    
    return [M]
    
def GHZ_basis_measurementv2():
    ket000 = np.array([1,0,0,0,0,0,0,0])
    ket111 = np.array([0,0,0,0,0,0,0,1])
    ket001 = np.array([0,1,0,0,0,0,0,0])
    ket010 = np.array([0,0,1,0,0,0,0,0])
    ket100 = np.array([0,0,0,0,1,0,0,0])    
    ket011 = np.array([0,0,0,1,0,0,0,0])
    ket101 = np.array([0,0,0,0,0,1,0,0])
    ket110 = np.array([0,0,0,0,0,0,1,0])
    M1 = (ket000 + ket111)*1/np.sqrt(2)
    M2 = (ket000 - ket111)*1/np.sqrt(2)
    M3 = (ket001 + ket110)*1/np.sqrt(2)
    M4 = (ket001 - ket110)*1/np.sqrt(2)
    M5 = (ket010 + ket101)*1/np.sqrt(2)
    M6 = (ket010 - ket101)*1/np.sqrt(2)
    M7 = (ket100 + ket011)*1/np.sqrt(2)
    M8 = (ket100 - ket011)*1/np.sqrt(2)
    
    Mout = []
    for Mi in [M1, M2, M3, M4, M5, M6, M7, M8]:
        Mi = np.reshape(Mi, [1,8]).astype("complex128")
        Mout.append(Mi*hconj(Mi))
        
    return [Mout]
   
# def TCinspired3parties():
#     ket000 = np.array([1,0,0,0,0,0,0,0])
#     ket111 = np.array([0,0,0,0,0,0,0,1])
#     ket001 = np.array([0,1,0,0,0,0,0,0])
#     ket010 = np.array([0,0,1,0,0,0,0,0])
#     ket100 = np.array([0,0,0,0,1,0,0,0])    
#     ket011 = np.array([0,0,0,1,0,0,0,0])
#     ket101 = np.array([0,0,0,0,0,1,0,0])
#     ket110 = np.array([0,0,0,0,0,0,1,0])
#     w=np.exp(1j*2*np.pi/3)
    
#     M0 = ket000
#     M1 = ket111
#     M2 = 1/np.sqrt(3) * (1 * ket001 + w * ket010 + w**2 * ket100)
#     M3 = 1/np.sqrt(3) * (w * ket001 + w**2 * ket010 + 1 * ket100)
#     M4 = 1/np.sqrt(3) * (w**2 * ket001 + 1 * ket010 + w * ket100)
#     #find three other orthogonal states (should it be combination of ket011 ket 101 ket 110 ?)
#     M5 = 1/np.sqrt(3) * (1 * ket011 + w * ket101 + w**2 * ket110) #can this work ??
#     M6 = 1/np.sqrt(3) * (w * ket011 + w**2 * ket101 + 1 * ket110)
#     M7 = 1/np.sqrt(3) * (w**2 * ket011 + 1 * ket101 + w * ket110)
#     Mout = []
#     for Mi in [M0, M1, M2, M3, M4, M5, M6, M7]:
#         Mi = np.reshape(Mi, [1,8]).astype("complex128")
#         Mout.append(Mi*hconj(Mi))
#     return [Mout]
 
def TCinspired3parties():
    ket000 = np.array([1,0,0,0,0,0,0,0])
    ket111 = np.array([0,0,0,0,0,0,0,1])
    ket001 = np.array([0,1,0,0,0,0,0,0])
    ket010 = np.array([0,0,1,0,0,0,0,0])
    ket100 = np.array([0,0,0,0,1,0,0,0])    
    ket011 = np.array([0,0,0,1,0,0,0,0])
    ket101 = np.array([0,0,0,0,0,1,0,0])
    ket110 = np.array([0,0,0,0,0,0,1,0])
    w=np.exp(1j*2*np.pi/3)
    
    M0 = ket000
    M1 = ket111
    M2 = 1/np.sqrt(3) * (1 * ket001 + w * ket010 + w**2 * ket100)
    M3 = 1/np.sqrt(3) * (1 * ket001 + w**2 * ket010 + w * ket100)
    M4 = 1/np.sqrt(3) * (1 * ket001 + 1 * ket010 + 1 * ket100)
    M5 = 1/np.sqrt(3) * (1 * ket011 + w * ket101 + w**2 * ket110)
    M6 = 1/np.sqrt(3) * (1 * ket011 + w**2 * ket101 + w * ket110)
    M7 = 1/np.sqrt(3) * (1 * ket011 + 1* ket101 + 1 * ket110)
    Mout = []
    for Mi in [M0, M1, M2, M3, M4, M5, M6, M7]:
        Mi = np.reshape(Mi, [1,8]).astype("complex128")
        Mout.append(Mi*hconj(Mi))
    return [Mout]
    
def get_measurement(i):
    if i==0: #standard elegant measurement
        return elegant_family(0)
    elif i==1: #measurment inbetween elegant and BSM
        return elegant_family(np.pi/4)
    elif i==2: #standard BSM measurement
        return elegant_family(np.pi/2)
    elif i==3: #TC measurement with u = sqrt(0.63)
        return TCfamily(np.sqrt(0.63))
    elif i==4: #TC measurement with u = sqrt(0.785)
        return TCfamily(np.sqrt(0.785))
    elif i==5: #TC measurement with u = sqrt(0.85)
        return TCfamily(np.sqrt(0.85))
    else:
        raise(ValueError("input should be in {0, ..., 5}"))
        
def get_measurement2(n, i):
    """n is the number of systems (2 or 3) and i is the id of the measurement"""
    if n==2:
        if i==0: #standard elegant measurement
            return elegant_family(0)
        elif i==1: #measurment inbetween elegant and BSM
            return elegant_family(np.pi/4)
        elif i==2: #standard BSM measurement
            return elegant_family(np.pi/2)
        elif i==3: #TC measurement with u = sqrt(0.63)
            return TCfamily(np.sqrt(0.63))
        elif i==4: #TC measurement with u = sqrt(0.785)
            return TCfamily(np.sqrt(0.785))
        elif i==5: #TC measurement with u = sqrt(0.85)
            return TCfamily(np.sqrt(0.85))
        else:
            raise(ValueError("input should be in {0, ..., 5}"))
    elif n==3:
        if i==0:
            return GHZ_basis_measurementv2()
        elif i==1:
            return TCinspired3parties()
        else:
            raise(ValueError("input for 3party measurment should be in {0,1}"))
    else:
        raise(ValueError("2 or 3 party measurement only"))
            
if __name__=="__main__":
    ## test that this measurement sum to identity
    Mout = TCinspired3parties()
    s=np.zeros([8,8]).astype("complex128")
    for i in range(8):
        s+=Mout[0][i]
    print(s)
    