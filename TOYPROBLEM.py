# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr, spsolve
from matplotlib import pyplot as plt
import time
import main_rows as mr

np.random.seed(1)

data = [1, 100, 500, 50, 50]

# Number of DoFs
nn = data[1]

# Number of DoFs - Number of secondary DoFs (= internal + main DoFs)
nt = nn-int(2*np.sqrt(nn))

# Creation of the matrices
M = np.identity(nn)

v_1 = np.ones(nn-1)
v_2 = -2*np.ones(nn)
Lap = np.diag(v_1,-1)+np.diag(v_2,0)+np.diag(v_1,1)

# Creation of the MPC matrix
case = data[0]
if case == 1:
	# CASE 1 
	T = np.zeros((nn,nt))
	for i in range(0,nt):
		T[i,i] = 1
	for i in range(nt, nn):
		T[i, i-nt] = 1
	np.random.shuffle(T)
else:
	# CASE 2   
	T = np.zeros((nn,nt))
	for i in range(0,nt):
		T[i,i] = 1
	for i in range(nt, nn):
		index = np.random.choice(np.arange(0, nt), size = 3, replace = False)
		T[i,index[0]] = 1/np.random.choice(np.arange(1,10))
		T[i,index[1]] = 1/np.random.choice(np.arange(1,10))
		T[i,index[2]] = 1/np.random.choice(np.arange(1,10))
	np.random.shuffle(T)
T = csr_matrix(T)

main, order = mr.find_main_rows(T)
secondary = [i for i in range(nn) if i not in main]

# Vector for the MPC application
b = np.zeros(nn)
for i in secondary:
    b[i] = i

# Time discretization
nsteps = data[2]
t_0 = 0
t_end = 5
delta_t = (t_end-t_0)/nsteps
print(f"\nThe interval of time is: {delta_t}")

# FOM system
K = M + delta_t*Lap
K = csr_matrix(K)
M = csr_matrix(M)

# Congruential transformation of the system
K_hat = T.T @ K @ T

# Solutions
snapshots = np.zeros((nn,nsteps))

# Initial condition
u_0 = np.zeros(nn)
for i in range(0,nn):
    u_0[i] = i
for i in secondary:
	u_0[i] += b[i]
u = u_0

# FOM solution
ti_FOM = time.time()
for i in range(0,nsteps):
    f = M @ u - K @ b
    f_hat = T.T @ f 
    u_hat = spsolve(K_hat, f_hat)
    u = T @ u_hat + b 
    snapshots[:,i] = u

time_FOM = time.time() - ti_FOM
print(f"\nThe time to solve the FOM system is: {time_FOM}")

# POD, computation of the SVD
ti_SVD = time.time()
U, S, Vt = linalg.svd(snapshots, full_matrices=False)
norm_S = linalg.norm(snapshots)
    
k = data[3] # To fix a value
 
Phi = U[:,:k]
print(f"\nThe basis shape for X is: {np.shape(Phi)}")
time_SVD = time.time() - ti_SVD
print(f"The time for the computation of the basis is: {time_SVD}")

# FULL SNAPSHOTS matrix
print(f"\nFull Snapshots matrix")
# Four approaches:
#   - QR decomposition
#   - PseudoInverse of T
#   - Least Square solution
#   - Main Reordered Rows of the basis

Kb = K @ b # used in more than one approaches 

# Approach 1: QR decomposition
sol_1 = np.zeros((nn,nsteps))
ti_1 = time.time()
Q, R = np.linalg.qr(T.toarray())
Rinv = np.linalg.pinv(R)
m_F1 = Phi.T @ Q @ Rinv.T @ T.T 
K_1 = m_F1 @ K @ m_F1.T
K_1 = csr_matrix(K_1)
m_U1 = Rinv @ Q.T @ Phi
b_U1 = Rinv @ Q.T @ b 
b_F1 = m_F1 @ K @ T @ b_U1

u = u_0
for i in range(0,nsteps):
	f = M @ u - Kb
	f_1 = m_F1 @ f + b_F1
	q = spsolve(K_1, f_1)
	u_hat = m_U1 @ q - b_U1
	u = T @ u_hat + b
	sol_1[:,i] = u
  
time_sol1 = time.time() - ti_1
norm_1 = linalg.norm(snapshots-sol_1)/norm_S
print(f"\nAPPROACH 1")
print(f"The norm of the error with the FOM simulation is: {norm_1}")
print(f"The time for the ROM simulation is: {time_sol1}")

del sol_1

# Approach 2: PseudoInverse of T
sol_2 = np.zeros((nn,nsteps))
ti_2 = time.time()
Tinv = linalg.pinv(T.toarray())
m_F2 = Phi.T @ Tinv.T @ T.T
K_2 = m_F2 @ K @ m_F2.T
K_2 = csr_matrix(K_2)
m_U2 = Tinv @ Phi
b_U2 = Tinv @ b
b_F2 = m_F2 @ K @ T @ b_U2

u = u_0
for i in range(0,nsteps):
	f = M @ u - Kb
	f_2 = m_F2 @ f + b_F2
	q = spsolve(K_2, f_2)
	u_hat = m_U2 @ q - b_U2
	u = T @ u_hat + b
	sol_2[:,i] = u

time_sol2 = time.time() - ti_2
norm_2 = linalg.norm(snapshots - sol_2)/norm_S
print(f"\nAPPROACH 2")
print(f"The norm of the error with the FOM simulation is: {norm_2}")
print(f"The time for the ROM simulation is: {time_sol2}")

del sol_2

# Approach 3: Least Square solution
sol_3 = np.zeros((nn,nsteps))
ti_3 = time.time()
# Least square solution
LS_Phi = np.empty((nt,k))
for i in range(0,k):
	LS_Phi[:,i] = lsqr(T, Phi[:,i])[0]
LS_b = lsqr(T, b)[0]
KTLS_b = K @ T @ LS_b
TLS_Phi = T @ LS_Phi
K_3 = TLS_Phi.T @ K @ TLS_Phi
K_3 = csr_matrix(K_3)

u = u_0
for i in range(0,nsteps):
	f = M @ u - Kb + KTLS_b
	f_3 = TLS_Phi.T @ f
	q = spsolve(K_3, f_3)
	u_hat = LS_Phi @ q - LS_b
	u = T @ u_hat + b
	sol_3[:,i] = u

time_sol3 = time.time() - ti_3
norm_3 = linalg.norm(snapshots - sol_3)/norm_S
print(f"\nAPPROACH 3")
print(f"The norm of the error with the FOM simulation is: {norm_3}")
print(f"The time for the ROM simulation is: {time_sol3}")

del sol_3

# Approach 4: Main Rows of the basis
sol_4 = np.zeros((nn,nsteps))
ti_4 = time.time()
MR_Phi = np.empty((nt,k))

MR_Phi = Phi[main,:]
MR_Phi = MR_Phi[order]
MR_b = b[main]
MR_b = MR_b[order]
KTMR_b = K @ T @ MR_b
TMR_Phi = T @ MR_Phi
K_4 = TMR_Phi.T @ K @ TMR_Phi
K_4 = csr_matrix(K_4)

u = u_0
for i in range(0,nsteps):
	f = M @ u - Kb + KTMR_b
	f_4 = TMR_Phi.T @ f
	q = spsolve(K_4, f_4)
	u_hat = MR_Phi @ q - MR_b
	u = T @ u_hat + b
	sol_4[:,i] = u

time_sol4 = time.time() - ti_4
norm_4 = linalg.norm(snapshots - sol_4)/norm_S
print(f"\nAPPROACH 4")
print(f"The norm of the error with the FOM simulation is: {norm_4}")
print(f"The time for the ROM simulation is: {time_sol4}")

del sol_4

# REDUCED SNAPSHOTS matrix (Only "main rows" of the snapshots mastrix)
print(f"\nReduced Snapshots matrix")

snapshots_t = snapshots[main, :] - b[main, np.newaxis]
snapshots_t = snapshots_t[order]

# POD, computation of the SVD
ti_SVD_t = time.time()
U_t, S_t, Vt_t = linalg.svd(snapshots_t, full_matrices=False)

k = data[4] # To fix a value
 
Phi_t = U_t[:,:k]
print(f"\nThe basis shape for X is: {np.shape(Phi_t)}")
time_SVD_t = time.time() - ti_SVD_t
print(f"The time for the computation of the basis is: {time_SVD_t}")

sol_5 = np.zeros((nn,nsteps))
ti_5 = time.time()

TPhi_t = T @ Phi_t
K_5 = TPhi_t.T @ K @ TPhi_t
K_5 = csr_matrix(K_5)

u = u_0
for i in range(0,nsteps):
	f = M @ u - Kb
	f_5 = TPhi_t.T @ f
	q = spsolve(K_5, f_5)
	u = TPhi_t @ q + b
	sol_5[:,i] = u

time_sol5 = time.time() - ti_5
norm_5 = linalg.norm(snapshots - sol_5)/norm_S
print(f"\nThe norm of the error with the FOM simulation is: {norm_5}")
print(f"The time for the ROM simulation is: {time_sol5}")
