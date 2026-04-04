# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import linalg
from scipy.sparse import diags, csr_matrix, identity
from scipy.sparse.linalg import spsolve, lsqr
import time
import main_rows as mr

n = 100
N = n*n
Domain = 1.0
Time = 5.0
N_ts = int(4/3*N)
Delta_t = Time/N_ts
h = Domain/(n-1)
D = 1

x = np.linspace(0, Domain, n)
y = np.linspace(0, Domain, n)
X, Y = np.meshgrid(x, y)
Xf = X.flatten()
Yf = Y.flatten()

v_1 = -4 * np.ones(N)
v_2 = np.ones(N - 1)
v_2 = np.ones(N - 1)
v_2[np.arange(1, n) * n - 1] = 0
v_3 = np.ones(N - n)
L = (diags([v_1, v_2, v_2, v_3, v_3], [0, -1, 1, -n, n], shape=(N, N)) / h**2)

M = identity(N)

K = (M - Delta_t * 1/D * L).tocsr()

boundary = []
for i in range(N):
    if Xf[i] == 0 or Xf[i] == 1.0 or Yf[i] == 0 or Yf[i] == 1.0:
        boundary.append(i)

l_boundary = [i for i in boundary if Xf[i] == 0]
r_boundary = [i for i in boundary if Xf[i] == 1.0]
u_boundary = [i for i in boundary if Yf[i] == 1.0]
d_boundary = [i for i in boundary if Yf[i] == 0]

P = 1e6
for i in l_boundary:
    K[i,i] = P
for i in d_boundary:
    K[i,i] = P

A = 10
def f_bottom(x, t):
    return A * (np.sin(2*np.pi*x) + np.sin(2*np.pi*t))
def f_left(y, t):
    return A * (np.sin(2*np.pi*y) + np.cos(2*np.pi*t))

secondary = r_boundary[1:].copy()
secondary.extend(u_boundary[1:-1])
secondary.sort()

main = [i for i in range(N) if i not in secondary]
Nt = len(main)

T = np.zeros((N,Nt))
for i in range(len(main)):
    T[main[i],i] = 1
for i in secondary[:-1]:
    if i in r_boundary:
        T[i, :] = T[i - n + 1, :]
    else:
        T[i, :] = T[main[u_boundary.index(i)], :]
T[-1,n - 1] = 0.5
T[-1,-1] = 0.5
T = csr_matrix(T)

u0 = np.zeros(N)
u_cond = u0.flatten()

for i in d_boundary:
    x_i = Xf[i]
    u_cond[i] = f_bottom(x_i, 0)
for i in l_boundary:
    y_i = Yf[i]
    u_cond[i] = f_left(y_i, 0)

for i in secondary[:-1]:
    if i in r_boundary:
        u_cond[i] = u_cond[i - n + 1]
    else:
        u_cond[i] = u_cond[main[u_boundary.index(i)]]
u_cond[-1] = 0.5 * u_cond[main[n-1]] + 0.5 * u_cond[main[-1]] 

b = np.zeros(N)
for i in secondary:
    b[i] = 15

snapshots = np.zeros((N,N_ts))

K_hat = T.T @ K @ T
u = u_cond
ti_FOM = time.time()
for i in range(N_ts):
    t = i * Delta_t

    g = np.zeros(N)
    g[d_boundary] = f_bottom(Xf[d_boundary], t)
    g[l_boundary] = f_left(Yf[l_boundary], t)
    
    f = M @ u - K @ b + P*g
    f_hat = T.T @ f 
    u_hat = spsolve(K_hat, f_hat)
    u = T @ u_hat + b
    snapshots[:,i] = u

time_FOM = time.time() - ti_FOM
print(f"\nThe time to solve the FOM system is: {time_FOM}")
np.save(f"snapshots_3.npy", snapshots)

# POD, computation of the SVD
ti_SVD = time.time()
U, S, Vt = linalg.svd(snapshots, full_matrices=False)

ControlNumberModes = True
norm_S = linalg.norm(snapshots)
if ControlNumberModes == True:
	m = np.size(S)
	k = m
	truncation_tolerance = 1e-12
	for t_1 in range(1,m):
		numerator = 0
		denominator = 0
		for t_2 in range(t_1,m):
			numerator += S[t_2]**2
		for t_3 in range (0,t_1):
			denominator += S[t_3]**2
		if np.sqrt(numerator/denominator) <= (truncation_tolerance * norm_S):
			k = t_1
			break

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
#   - Main Rows of the basis

Kb = K @ b # used in more than one approaches 

# Approach 1: QR decomposition
sol_1 = np.zeros((N,N_ts))
ti_1 = time.time()
Q, R = np.linalg.qr(T.toarray())
Rinv = np.linalg.pinv(R)
m_F1 = Phi.T @ Q @ Rinv.T @ T.T 
K_1 = m_F1 @ K @ m_F1.T
K_1 = csr_matrix(K_1)
m_U1 = Rinv @ Q.T @ Phi
b_U1 = Rinv @ Q.T @ b 
b_F1 = m_F1 @ K @ T @ b_U1

u = u_cond
for i in range(0,N_ts):
    t = i * Delta_t

    g = np.zeros(N)
    g[d_boundary] = f_bottom(Xf[d_boundary], t)
    g[l_boundary] = f_left(Yf[l_boundary], t)
    
    f = M @ u - K @ b + P*g
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

np.save(f"A1_3.npy", sol_1)
del sol_1

# Approach 2: PseudoInverse of T
sol_2 = np.zeros((N,N_ts))
ti_2 = time.time()
Tinv = linalg.pinv(T.toarray())
m_F2 = Phi.T @ Tinv.T @ T.T
K_2 = m_F2 @ K @ m_F2.T
K_2 = csr_matrix(K_2)
m_U2 = Tinv @ Phi
b_U2 = Tinv @ b
b_F2 = m_F2 @ K @ T @ b_U2

u = u_cond
for i in range(0,N_ts):
    t = i * Delta_t

    g = np.zeros(N)
    g[d_boundary] = f_bottom(Xf[d_boundary], t)
    g[l_boundary] = f_left(Yf[l_boundary], t)

    f = M @ u - Kb + P*g
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

np.save(f"A2_3.npy", sol_2)
del sol_2

# Approach 3: Least Square solution
sol_3 = np.zeros((N,N_ts))
ti_3 = time.time()
# Least square solution
LS_Phi = np.empty((Nt,k))
for i in range(0,k):
	LS_Phi[:,i] = lsqr(T, Phi[:,i])[0]
LS_b = lsqr(T, b)[0]
KTLS_b = K @ T @ LS_b
TLS_Phi = T @ LS_Phi
K_3 = TLS_Phi.T @ K @ TLS_Phi
K_3 = csr_matrix(K_3)

u = u_cond
for i in range(0,N_ts):
    t = i * Delta_t

    g = np.zeros(N)
    g[d_boundary] = f_bottom(Xf[d_boundary], t)
    g[l_boundary] = f_left(Yf[l_boundary], t)

    f = M @ u - Kb + KTLS_b + P*g
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

np.save(f"A3_3.npy", sol_3)
del sol_3

_, order = mr.find_main_rows(T)

# Approach 4: Main Rows of the basis
sol_4 = np.zeros((N,N_ts))
ti_4 = time.time()
MR_Phi = np.empty((Nt,k))

MR_Phi = Phi[main,:]
MR_Phi = MR_Phi[order]
MR_b = b[main]
MR_b = MR_b[order]
KTMR_b = K @ T @ MR_b
TMR_Phi = T @ MR_Phi
K_4 = TMR_Phi.T @ K @ TMR_Phi
K_4 = csr_matrix(K_4)

u = u_cond
for i in range(0,N_ts):
    t = i * Delta_t

    g = np.zeros(N)
    g[d_boundary] = f_bottom(Xf[d_boundary], t)
    g[l_boundary] = f_left(Yf[l_boundary], t)

    f = M @ u - Kb + KTMR_b + P*g
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

np.save(f"A4_3.npy", sol_4)
del sol_4

# REDUCED SNAPSHOTS matrix (Only "main rows" of the snapshots mastrix)
print(f"\nReduced Snapshots matrix")

snapshots_t = snapshots[main, :] - b[main, np.newaxis]
snapshots_t = snapshots_t[order]

# POD, computation of the SVD
ti_SVD_t = time.time()
U_t, S_t, Vt_t = linalg.svd(snapshots_t, full_matrices=False)
   
# Computation of the basis for the POD
norm_S_t = linalg.norm(snapshots_t)
m = np.size(S_t)
k = m
truncation_tolerance_t = 1e-12
for t_1 in range(1,m):
	numerator = 0
	denominator = 0
	for t_2 in range(t_1,m):
		numerator += S_t[t_2]**2
	for t_3 in range (0,t_1):
		denominator += S_t[t_3]**2
	if np.sqrt(numerator/denominator) <= (truncation_tolerance_t * norm_S_t):
		k = t_1
		break

Phi_t = U_t[:,:k]
print(f"\nThe basis shape for X is: {np.shape(Phi_t)}")
time_SVD_t = time.time() - ti_SVD_t
print(f"The time for the computation of the basis is: {time_SVD_t}")

sol_5 = np.zeros((N,N_ts))
ti_5 = time.time()

TPhi_t = T @ Phi_t
K_5 = TPhi_t.T @ K @ TPhi_t
K_5 = csr_matrix(K_5)

u = u_cond
for i in range(0,N_ts):
    t = i * Delta_t

    g = np.zeros(N)
    g[d_boundary] = f_bottom(Xf[d_boundary], t)
    g[l_boundary] = f_left(Yf[l_boundary], t)

    f = M @ u - Kb + P*g
    f_5 = TPhi_t.T @ f
    q = spsolve(K_5, f_5)
    u = TPhi_t @ q + b
    sol_5[:,i] = u

time_sol5 = time.time() - ti_5
norm_5 = linalg.norm(snapshots - sol_5)/norm_S
print(f"\nThe norm of the error with the FOM simulation is: {norm_5}")
print(f"The time for the ROM simulation is: {time_sol5}")

np.save(f"A5_3.npy", sol_5)
del sol_5

#fig, axes = plt.subplots(2, 3, figsize=(14, 9))
#snap_list = [snapshots, sol_1, sol_2, sol_3, sol_4, sol_5]
#titles = ["FOM", "QR", "Pseudo-Inverse", "Least Square", "MRR Basis", "MRR Snapshots"]
#
#vmin = min(s.min() for s in snap_list)
#vmax = max(s.max() for s in snap_list)
#
#snap_list_r = [s.reshape((n, n, -1)) for s in snap_list]
#
#ims = []
#for ax, snap, title in zip(axes.flat, snap_list_r, titles):
#    im = ax.imshow(snap[:, :, 0],
#                   origin='lower', extent=[0, Domain, 0, Domain],
#                   cmap='jet', vmin=vmin, vmax=vmax,
#                   interpolation='bilinear',
#                   aspect='auto')
#    ax.set_title(title, fontsize=12)
#    ims.append(im)
#
#fig.colorbar(ims[0], cax=cbar_ax)
#
#suptitle = fig.suptitle("Time Step: 0, t = 0 s", fontsize=14, y=0.98)
#
#def update(frame):
#    suptitle.set_text(f"Time Step: {frame}, t = {frame*Delta_t:.2f} s")
#    for im, snap in zip(ims, snap_list_r):
#        im.set_data(snap[:, :, frame])
#    return ims
#
#step = 2
#frames = range(0, N_ts, step)
#
#anim = FuncAnimation(fig, update, frames, interval=1, blit=False)
#plt.show()
#anim.save("VideoTest.mp4", fps=30, dpi=150)