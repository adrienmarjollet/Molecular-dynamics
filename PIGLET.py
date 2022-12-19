import numpy as np
from numpy import linalg as LA
import os

current_directory = os.getcwd()

'''Convertion units and conventions'''
au2cm = 27.211 * 8065.5
au2K = 315774.641
kcalmol2cm = 349.75
fs2au = 41.341374575

'''
WE TREAT THE 1D CASE / QT. 
'''
Ns = 1

Aqp, Bqp, Cqp = np.zeros(
    (2+Ns, 2+Ns)), np.zeros((2+Ns, 2+Ns)), np.zeros((2+Ns, 2+Ns))
Ap, Bp, Cq = np.zeros((1+Ns, 1+Ns)), np.zeros((1+Ns, 1+Ns)
                                              ), np.zeros((1+Ns, 1+Ns))

'''
FROM EOM:
               Aqp Cpq + Cqp Aqp.T  =  Bp Bp.T    (1)
'''

'''
WELL BEHAVED SDE -> Constraints on the Memory kernel K -> Constraints on A: 
                   K with exponential decay      -> Re(Spec(Ap)) > 0                  (i)
                   Second Law thermo consistency -> K(w) > 0  -> Ap+Ap.T posi. defi.  (ii)
                   
                   -> 2n+1 + n(n+1)/2 free parameters
'''

'''
Bp : REAL lower triangular (iii)
                
             -> (n+1)(n+2)/2 free parameters                           
'''

'''
(i) + (ii) (TO BE CHECKED)
'''
ASp, AAp = np.zeros((1+Ns, 1+Ns)), np.zeros((1+Ns, 1+Ns))
Qp = np.zeros((1+Ns, 1+Ns))


print('Number of free parameters to fit with Ns='+str(Ns) +
      ': \n',  2*(Ns+1)+1+(Ns+1)*(Ns+2)/2+(Ns+2)*(Ns+3)/2)
print()

'''
IN THE HARMONIC  REGIME FOR WHICH EVERYTHING IS ANALYTIC WE CAN WORK WITH ONLY the p part of the matrices

BY USING (1) WE CAN GET Cp FROM Ap and Bp
'''


def chifit4(m, w_arr, tCqq_arr, tCpp_arr, Cqq_arr, Cpp_arr):
    S = float(0)
    for i in range(len(w_arr)):
        S += abs(np.log(Cqq_arr[i]/tCqq_arr[i]))**m + \
            abs(np.log(Cpp_arr[i]/tCpp_arr[i]))**m
    S = S**(1.0/m)
    return S


def cov_elem(n, i, j, d_arr, P, Mfact):
    S = float(0)
    for k in range(n):
        for l in range(n):
            S += (P[i, k] * Mfact[k, l] * P[j, l]) / (d_arr[k] + d_arr[l])
    return S


def cov_mat(Ns, Aqp, Bqp):
    '''
    Derive covariance matrix from (1), uses a trick
    '''
    Cqp = np.zeros((Ns+2, Ns+2))
    '''
    eigendecomposition of Ap
    '''
    d_arr, P = LA.eig(Aqp)
    DAqp = np.diag(d_arr)
    Pinv = LA.inv(P)
    BT = Bqp.transpose()
    PinvT = Pinv.transpose()
    Mfact = np.dot(np.dot(np.dot(Pinv, B), BT), PinvT)
    for i in range(len(Cqp)):
        for j in range(len(Cqp[0])):
            Cqp[i, j] = cov_elem(Ns+2, i, j, d_arr, P, Mfact)
    return Cqp


def drift_mat(Ns, Ap, Cp):
    # TODO
    return Bp


'''
FUNCTIONS FOR HARMONIC REGIME (w)
'''


def coth(x):
    return np.cosh(x)/np.sinh(x)


def harmo_AB(Ns, w, Ap, Bp):
    '''
    RETURN Apq,Bpq,Cpq
    (Improve with fancy indexing ?)
    '''
    Aqp, Bqp, Cqp = np.zeros(
        (2+Ns, 2+Ns)), np.zeros((2+Ns, 2+Ns)), np.zeros((2+Ns, 2+Ns))
    for i in range(1, Ns+2):
        for j in range(1, Ns+2):
            Apq[i, j] = Ap[i, j]
            Bpq[i, j] = Ap[i, j]

    Apq[1, 0], Apq[0, 1] = w**2, -1.

    Cqp = cov_mat(Ns, Aqp, Bqp)

    return Apq, Bpq, Cpq


def tqp_fluctharmo(w, m, beta, v):
    if v == 0:
        coef = 1.0
    elif v == 1:
        coef = 3.0
    tdqq = coef * (1.0/(2.0*w*m)) * coth(w*beta/2.0)
    tdpp = coef * (w*m/2.0) * coth(w*beta/2.0)
    return tdqq, tdpp


def qp_fluctharmo(Ns, Aqp, Bqp):
    Cqp = cov_mat(Ns, Aqp, Bqp)
    dqq = Cqp[0, 0]
    dpp = Cqp[1, 1]
    return dqq, dpp


'''
AN: TEST FOR v=0 analytically!
    m = 1
    gle4md
'''
Ns = 6
m = 1
beta = au2K/500.0
Eth = 1/beta

'''
hw/kT =20
'''
w = 20*Eth


A = np.array([9.443659190019e-6,    1.132294186893e-3,    7.449628436962e-4,    1.071101169467e-3,    1.223429397683e-3,    6.267797080926e-4,    3.661966178600e-4,
              -1.158486730163e-3,    4.928600223455e-4,    1.595931933797e-3,   -
              2.062890689123e-4,   -5.396142818401e-5,    1.594791295865e-4,   -1.609441915443e-3,
              -7.027092008132e-4,   -1.595931933797e-3,    5.047934082853e-4,    2.770764198362e-4,    7.526274670274e-4,    2.664551238983e-4,   -7.897558467954e-4,
              -1.072792654821e-3,    2.062890689123e-4,   -
              2.770764198362e-4,    5.772022515510e-4,    2.126703161982e-5,    7.244407882208e-4,    3.701736742286e-4,
              -1.189347224241e-3,    5.396142818401e-5,   -7.526274670274e-4,   -
              2.126703161982e-5,    6.304868735244e-4,   -
              8.366171930319e-4,    2.357048551293e-3,
              -3.148986705340e-4,   -1.594791295865e-4,   -2.664551238983e-4,   -
              7.244407882208e-4,    8.366171930319e-4,    5.722269541087e-3,   -7.870057326856e-3,
              -7.265098725829e-4,    1.609441915443e-3,    7.897558467954e-4,   -
              3.701736742286e-4,   -2.357048551293e-3,    7.870057326856e-3,    9.273239517793e-3
              ])

C = np.array([1.583400271683e-3,  3.133877015476e-5, 1.131807557815e-3, 3.105092020356e-4,  -5.910353041934e-4, -3.585564345555e-4,  7.282565650989e-5,
              3.133877015476e-5,  1.996070832363e-3, -3.058361760595e-4, -
              1.149961818023e-3, -8.705104025437e-4, -3.460861323567e-4,  2.023505714317e-4,
              1.131807557815e-3,   -3.058361760595e-4, 3.866554938147e-3, 1.155099992656e-3, -
              1.955290287227e-3, -1.786714017672e-4,  -2.548076028056e-5,
              3.105092020356e-4,   -1.149961818023e-3, 1.155099992656e-3,  3.673733635881e-3,  -
              1.916013627896e-3,  8.988166263167e-4,  -3.205485344848e-4,
              -5.910353041934e-4,   -8.705104025437e-4, -1.955290287227e-3, -
              1.916013627896e-3,  7.589126908389e-3, 5.936445689760e-4,   1.865124585796e-5,
              -3.585564345555e-4,   -3.460861323567e-4,  -
              1.786714017672e-4,   8.988166263167e-4,   5.936445689760e-4, 7.860770714961e-3,  -
              8.484963665908e-4,
              7.282565650989e-5,  2.023505714317e-4, -2.548076028056e-5,   -3.205485344848e-4,  1.865124585796e-5, -8.484963665908e-4, 2.346126568789e-3])

print('beta=', beta)
print('1/beta=', Eth)
print('w(cm^-1)=', w*au2cm)
# print(A.shape)
A = A.reshape((Ns+1, Ns+1))
C = C.reshape((Ns+1, Ns+1))
# print(A.reshape((Ns+1,Ns+1)))
print(A.shape)


dummy, tdpp = tqp_fluctharmo(w, m, beta, 0)

print('tdpp:', tdpp)
print('C[0,0]:', C[0, 0])
