# numerico.py: Library contaning numeric methods
# Author: Luis Vasquez
# Computer Science
# Universidad Nacional de Ingenieria - Lima, Peru

import numpy as np

# Epsilon_Mach
#   return: Machine epsilon
def Epsilon_Mach():
	eps = 1.0
	while eps + 1.0 > 1.0:
		epsilon = eps
		eps = 0.5*eps
	return epsilon

# Pivot_mat
    # A: Matrix for pivoting
    # c_index: column index (generally) where we apply de pivoting

    # return: permutation matrix 'P'
def Pivot_mat(A, c_index):
    dim = A.shape[0]
    P = np.eye(dim)

    index = c_index
    for i in range(c_index + 1, dim):
        if abs(A[i,c_index]) > abs(A[index,c_index]):
            index = i

    if index != c_index:
        P[[index, c_index]] = P[[c_index, index]]

    return P

# back_solve_triangular
    # A: Coef. matrix
    # b: solutions vector

    # return: solutions vector x from backward evaluations
def back_solve_triangular(A, b, verbose=False):
    x = np.empty(A.shape[1])
    rows = A.shape[0]
    for i in range(rows-1, -1, -1):
        pre_sum = A[i, i+1:].dot(x[i+1:])
        x[i] = (b[i] - pre_sum)/A[i,i]
        # ========================================
        if verbose:
            print("x_{}".format(i+1))
            print(x[i])
            print("========================")
        # ========================================
    # ========================================
    if verbose:
        print("x")
        print(x)
        print("========================")
    # ========================================
    return x

# for_solve_triangular
    # A: Coef. matrix
    # b: solutions vector

    # return: solutions vector x from forward evaluations
def for_solve_triangular(A,b, verbose=False):
    x = np.zeros(A.shape[1])
    rows = A.shape[0]
    for i in range(0, rows):
        if i == 0:
            x[0] = b[0]/A[0,0]
        else:
            pre_sum = A[i,0:i].dot(x[0:i])
            x[i] = (b[i] - pre_sum)/A[i,i]
        # ========================================
        if verbose:
            print("x_{}".format(i+1))
            print(x[i])
            print("========================")
        # ========================================
    # ========================================
    if verbose:
        print("x")
        print(x)
        print("========================")
    # ========================================
    return x

def solve_tridiagonal(A, b_0, verbose=False):
	a = np.copy(np.diag(A, k=-1))
	b = np.copy(np.diag(A))
	c = np.copy(np.diag(A, k=1))
	d = np.copy(b_0)

	n = len(d) # n is the numbers of rows, a and c has length n-1
	for i in range(n-1):
		d[i+1] -= 1. * d[i] * a[i] / b[i]
		b[i+1] -= 1. * c[i] * a[i] / b[i]
	for i in reversed(range(n-1)):
		d[i] -= d[i+1] * c[i] / b[i+1]
	return [d[i] / b[i] for i in range(n)]


# decompose
    # A: Matrix

    # return: 	D = Diagonal matrix from A,
	#			E: negative of l.tri. matrix from A (no diag),
	#			F: negative of u.tri. matrix from A (no diag)
def decompose(A, verbose=False):
	E = -np.tril(A, -1)
	F = -np.triu(A, 1)
	D = np.diag(A.diagonal())
	if verbose:
		print("D: \n", D)
		print("E: \n", E)
		print("F: \n", F)

	return D,E,F

# Gauss
    # A: Coef. matrix
    # b: Results matrix
    # verbose: printing flag
    # pivot: Pivoting criteria
    # lu: Asks if return L and U matrices from decomposition

    # return: X vector of solutions
def Gauss(A, b, verbose=False, pivot="none"):
    A_save = A
    A_b = np.matrix(np.c_[A, b]) # Augmented matrix
    n_row, n_col = A_b.shape[0], A_b.shape[1]
    process_string = "A_b"

    P = np.eye(n_row)
    L = np.eye(n_row)

    # ========================================
    if verbose:
        print("Matriz aumentada al inicio:")
        print(A_b)
        print("========================")
    # ========================================


    for i in range(0, n_row-1): # Main loop
        # Gauss matrices preparation
        P_i = np.eye(n_row)
        L_i = np.eye(n_row)

        if (A_b[i,i] == 0 and pivot == "partial") or pivot=="total":
            P_i = Pivot_mat(A_b, i)

        # ========================================
        if verbose:
            print("P_{}".format(i+1))
            print(P_i)
            print("========================")
        # ========================================

        # Apply permutation
        A_b = np.matmul(P_i,A_b)
        P = np.matmul(P_i, P)
        L = np.matmul(P_i, L)

        # ========================================
        if verbose:
            process_string = "P_{}.".format(i+1) + process_string
            print(process_string)
            print(A_b)
            print("========================")
        # ========================================

        # Preparing alpha_i
        alpha_i = np.zeros(n_row)
        alpha_i[i+1:] = np.transpose(A_b[i+1:n_row,i])/A_b[i,i]

        # ========================================
        if verbose:
            print("alpha_{}".format(i+1))
            print(alpha_i)
            print("========================")
        # ========================================

        # Preparing e_i
        e_i = np.zeros(n_row)
        e_i[i] = 1

        # Calc. L_i
        L_i = np.eye(n_row) - np.outer(alpha_i, e_i)
        L = np.matmul(L_i, L)

        # ========================================
        if verbose:
            print("L_{}".format(i+1))
            print(L_i)
            print("========================")
        # ========================================

        # Transform system
        A_b = np.matmul(L_i,A_b)
        # ========================================
        if verbose:
            process_string = "L_{}.".format(i+1) + process_string
            print(process_string)
            print(A_b)
            print("========================")
        # ========================================

    # Final calculus of LU matrices
    U = A_b[:,:n_col-1]
    L = np.linalg.inv(np.matmul(L, np.linalg.inv(P)))

    # ========================================
    if verbose:
        print("Ab final (Gauss)")
        print(A_b)
        print("========================")
        print("L final (Gauss)")
        print(L)
        print("========================")
        print("U final (Gauss)")
        print(U)
        print("========================")
    # ========================================

    return P, L, U

# Gauss-Jordan
    # A: Coef. matrix
    # b: Results matrix
    # verbose: printing flag
    # pivot: Pivoting criteria

    # return: X vector of solutions
def Gauss_Jordan(A, b, verbose=False, pivot="none"):
    A_b = np.matrix(np.c_[A, b]) # Augmented matrix
    n_row, n_col = A_b.shape[0], A_b.shape[1]
    process_string = "A_b"

    # ========================================
    if verbose:
        print("Matriz aumentada al inicio:")
        print(A_b)
        print("========================")
    # ========================================


    for i in range(0, n_row): # Main loop
        # Gauss matrices preparation
        P_i = np.eye(n_row)
        T_i = np.eye(n_row)

        if (A_b[i,i] == 0 and pivot == "partial") or pivot=="total":
            P_i = Pivot_mat(A_b, i)

        # ========================================
        if verbose:
            print("P_{}".format(i+1))
            print(P_i)
            print("========================")
        # ========================================

        # Apply permutation
        A_b = np.matmul(P_i,A_b)

        # ========================================
        if verbose:
            process_string = "P_{}.".format(i+1) + process_string
            print(process_string)
            print(A_b)
            print("========================")
        # ========================================

        # Preparing alpha_i
        alpha_i = np.empty(n_row)
        alpha_i[:] = (A_b[:,i].T/A_b[i,i])
        alpha_i[i] = 1/A_b[i,i]

        # ========================================
        if verbose:
            print("alpha_{}".format(i+1))
            print(alpha_i)
            print("========================")
        # ========================================

        # Preparing e_i
        e_i = np.zeros(n_row)
        e_i[i] = 1

        # Calc. L_i
        T_i = np.eye(n_row) - np.outer(alpha_i, e_i)

        # ========================================
        if verbose:
            print("T_{}".format(i+1))
            print(T_i)
            print("========================")
        # ========================================

        # Transform system
        A_b = np.matmul(T_i,A_b)
        # ========================================
        if verbose:
            process_string = "T_{}.".format(i+1) + process_string
            print(process_string)
            print(A_b)
            print("========================")
        # ========================================

    # ========================================
    if verbose:
        print("Ab final (Gauss-Jordan)")
        print(A_b)
        print("========================")
    # ========================================
    return back_solve_triangular(A_b[:,:n_col-1],A_b[:,n_col-1], verbose)

# Inverse
    # A: Matrix
    # verbose: printing flag

    # return: Matrix A^-1 inverse of A
def Inverse(A, verbose=False):
    n_row, n_col = A.shape[0], A.shape[1]
    if n_row != n_col or np.linalg.det(A) == 0:
        print("Singular matrix or dimensions error")
        return
    process_string = "A"
    cp = A
    A_inv = np.eye(n_row)

    # ========================================
    if verbose:
        print("Matriz al inicio:")
        print(A)
        print("========================")
    # ========================================


    for i in range(0, n_row): # Main loop
        # Gauss matrices preparation
        T_i = np.eye(n_row)

        # Preparing alpha_i
        alpha_i = np.empty(n_row)
        alpha_i[:] = (A[:,i].T/A[i,i])
        alpha_i[i] = 1/A[i,i]

        # Preparing e_i
        e_i = np.zeros(n_row)
        e_i[i] = 1

        # Calc. L_i
        T_i = np.eye(n_row) - np.outer(alpha_i, e_i)
        # Gauss-Jordan variation to calculate inverse, forcing T-tilde[i,i] = 1/A[i,i]
        T_i[i,i] = 1/A[i,i]

        # ========================================
        if verbose:
            print("T-tilde_{}".format(i+1))
            print(T_i)
            print("========================")
        # ========================================

        # Transform system
        A = np.matmul(T_i, A)
        A_inv = np.matmul(T_i, A_inv)
        # ========================================
        if verbose:
            process_string = "T-tilde_{}.".format(i+1) + process_string
            print(process_string)
            print(A)
            print("========================")
        # ========================================

    # ========================================
    if verbose:
        print("Resultado de la inversión:")
        print(A_inv)
        print("========================")
    # ========================================
    return A_inv

# LU Decomposition: Direct crout method 1
#   A: Matrix in which we apply LU_1 decomp.
#   return: P, L, U matrices
def LU_1(A, verbose=False, pivot="none"):
    dim = A.shape[0]

    if(dim != A.shape[0]):
        print("Error. 'A' matriz no cuadrada")
        return

    process_string = "A"

    # ========================================
    if verbose:
        print("Matriz A al principio")
        print(A)
        print("========================")
    # ========================================

    P = np.eye(dim)
    L = np.zeros((dim, dim))
    U = np.eye(dim)

    for k in range(0, dim):
        P_k = np.eye(dim)

        if (A[k,k] == 0 and pivot == "partial") or pivot=="total":
            P_k = Pivot_mat(A, k)

        # ========================================
        if verbose:
            print("P_{}".format(k+1))
            print(P_k)
            print("========================")
        # ========================================

        A = np.matmul(P_k, A)
        P = np.matmul(P_k, P)
        # ========================================
        if verbose:
            process_string = "P_{}.".format(k+1) + process_string
            print(process_string)
            print(A)
            print("========================")
        # ========================================

    for k in range(0, dim):
        # l_ik
        for i in range(k, dim): # rows iteration
            pre_sum = L[i,0:k].dot(U[0:k,k])
            L[i,k] = A[i,k] - pre_sum

        # ========================================
        if verbose:
            print("L_{}".format(k+1))
            print(L)
            print("========================")
        # ========================================

        # u_kj
        for j in range(k+1, dim): # cols iteration
            pre_sum = L[k,0:k].dot(U[0:k, j])
            U[k,j] = (A[k,j] - pre_sum)/L[k,k]

        # ========================================
        if verbose:
            print("U_{}".format(k+1))
            print(U)
            print("========================")
        # ========================================

    # ========================================
    if verbose:
        print("P final:")
        print(P)
        print("========================")
        print("L final:")
        print(L)
        print("========================")
        print("U final:")
        print(U)
        print("========================")
    # ========================================
    return P,L,U

# LU Decomposition: Direct crout method 2
#   A: Matrix in which we apply L_1U decomp.
#   return: P, L, U matrices
def L_1U(A, verbose=False, pivot="none"):
    dim = A.shape[0]

    if(dim != A.shape[0]):
        print("Error. 'A' matriz no cuadrada")
        return

    process_string = "A"

    # ========================================
    if verbose:
        print("Matriz A al principio")
        print(A)
        print("========================")
    # ========================================

    P = np.eye(dim)
    L = np.eye(dim)
    U = np.zeros((dim, dim))

    for k in range(0, dim):
        P_k = np.eye(dim)

        if (A[k,k] == 0 and pivot == "partial") or pivot=="total":
            P_k = Pivot_mat(A, k)

        # ========================================
        if verbose:
            print("P_{}".format(k+1))
            print(P_k)
            print("========================")
        # ========================================

        A = np.matmul(P_k, A)
        P = np.matmul(P_k, P)
        # ========================================
        if verbose:
            process_string = "P_{}.".format(k+1) + process_string
            print(process_string)
            print(A)
            print("========================")
        # ========================================

    for k in range(0, dim):
        # l_ik
        for j in range(k, dim): # rows iteration
            pre_sum = L[k,0:k].dot(U[0:k,j])
            U[k,j] = A[k,j] - pre_sum

        # ========================================
        if verbose:
            print("U_{}".format(k+1))
            print(U)
            print("========================")
        # ========================================

        # u_kj
        for i in range(k+1, dim): # cols iteration
            pre_sum = L[i,0:k].dot(U[0:k, k])
            L[i,k] = (A[i,k] - pre_sum)/U[k,k]

        # ========================================
        if verbose:
            print("L_{}".format(k+1))
            print(L)
            print("========================")
        # ========================================

    # ========================================
    if verbose:
        print("P final:")
        print(P)
        print("========================")
        print("L final:")
        print(L)
        print("========================")
        print("U final:")
        print(U)
        print("========================")
    # ========================================
    return P,L,U

# LDLt
#   A: Simetric matrix in which we apply LDLt decomp
#   verbose: printing flag
#   pivot: Pivoting criteria
#
#   return: P, L, D matrices such that PAPt = LDLt

def LDLt(A, verbose=False, pivot="none"):
	n = A.shape[0]
	P = np.eye(n)
	L = np.eye(n)
	D = np.zeros(n)
	process_string = "A"

	eig_values = np.linalg.eig(A)[0]
	for i in eig_values:
	    if i < 0:
	        print("(LDLt) Error: matriz no definida positiva")
	        return

	for k in range(0, n):
	    P_k = np.eye(n)

	    if (A[k,k] == 0 and pivot == "partial") or pivot=="total":
	        P_k = Pivot_mat(A, k)

	    # ========================================
	    if verbose:
	        print("P_{}".format(k+1))
	        print(P_k)
	        print("========================")
	    # ========================================

	    A = np.matmul(P_k, A)
	    A = np.matmul(A, P_k.T)
	    P = np.matmul(P_k, P)

	    # ========================================
	    if verbose:
	        process_string = "P_{}.".format(k+1) + process_string + ".P^T_{}".format(k+1)
	        print(process_string)
	        print(A)
	        print("========================")
	    # ========================================

	    pre_sum = 0
	    for v in range(0, k):
	        pre_sum += D[v]*(L[k, v])**2
	    D[k] = A[k,k] - pre_sum

	    for i in range(k+1, n):
	        pre_sum_2 = 0
	        for v in range(0, k):
	            pre_sum_2 += L[i, v]*D[v]*L[k, v]

	        L[i,k] = (A[i,k] - pre_sum_2)/D[k]

	    # ========================================
	    if verbose:
	        print("D_{}.".format(k+1))
	        print(D)
	        print("========================")
	    # ========================================

	    # ========================================
	    if verbose:
	        print("L_{}.".format(k+1))
	        print(L)
	        print("========================")
	    # ========================================

	# ========================================
	if verbose:
		print("P")
		print(P)
		print("========================")
		print("L")
		print(L)
		print("========================")
		print("D")
		print(np.diag(D))
		print("========================")
    # ========================================
	return P, L, np.diag(D)

# Cholesky
#   A: Simetric matrix in which we apply Cholesky decomp
#   verbose: printing flag
#   pivot: Pivoting criteria
#
#   return: G lower triangular matrix
def Cholesky(A, verbose=False, pivot="none"):
	eig_values = np.linalg.eig(A)[0]
	n = A.shape[0]
	G = np.zeros((n,n))
	P = np.eye(n)
	process_string = "A"

	for i in eig_values:
	    if i < 0:
	        print("(Cholesky) Error: matriz no definida positiva")
	        return

	for k in range(0, n):
		P_k = np.eye(n)

		if (A[k,k] == 0 and pivot == "partial") or pivot=="total":
			P_k = Pivot_mat(A, k)

		# ========================================
		if verbose:
		    print("P_{}".format(k+1))
		    print(P_k)
		    print("========================")
		# ========================================

		A = np.matmul(P_k, A)
		A = np.matmul(A, P_k.T)
		P = np.matmul(P_k, P)

		# ========================================
		if verbose:
			process_string = "P_{}.".format(k+1) + process_string + ".P^T_{}".format(k+1)
			print(process_string)
			print(A)
			print("========================")
	    # ========================================


		pre_sum = G[k, 0:k].dot(G[k, 0:k])
		G[k,k] = np.sqrt(A[k,k] - pre_sum)

		for i in range(k+1, n):
		    pre_sum_2 = G[i, 0:k].dot(G[k,0:k])
		    G[i,k] = (A[i,k] - pre_sum_2)/G[k,k]

		# ========================================
		if verbose:
		    print("G_{}.".format(k+1))
		    print(G)
		    print("========================")
# ========================================

	return P, G

# Pivot_mat_pr
    # A: Matrix for pivoting with Parlett-Reid criteria
    # c_index: column index (generally) where we apply de pivoting

    # return: permutation matrix 'P'
def Pivot_mat_pr(A, c_index):
    dim = A.shape[0]
    P = np.eye(dim)

    index = c_index + 1
    for i in range(c_index + 2, dim):
        if abs(A[i,c_index]) > abs(A[index,c_index]):
            index = i

    if index != c_index + 1:
        P[[index, c_index + 1]] = P[[c_index + 1, index]]

    return P

# Cholesky
#   A: Simetric matrix in which we apply Parlett & Reid decomp
#   verbose: printing flag
#   pivot: Pivoting criteria
#
#   return: P: permutation matrix, L: Lower triangular matrix, T: tridiagoinal matrix as PAP^t = LTL^t
def Parlett_Reid(A, verbose=False, pivot="none"):
    n_row, n_col = A.shape[0], A.shape[1]

    P = np.eye(n_row)
    L = np.eye(n_row)
    T = np.eye(n_row)

    process_string = "A"

    for k in range(0, n_row-2):
        P_k = np.eye(n_row)

        if (A[k,k] == 0 and pivot == "partial") or pivot=="total":
            P_k = Pivot_mat_pr(A, k)
        # ========================================
        if verbose:
            print("P_{}".format(k+1))
            print(P_k)
            print("========================")
        # ========================================
        A = np.matmul(P_k, A)
        A = np.matmul(A, P_k.T)

        P = np.matmul(P_k, P)
        L = np.matmul(P_k, L)

        # ========================================
        if verbose:
            process_string = "P_{}.".format(k+1) + process_string + ".P_{}^T".format(k+1)
            print(process_string)
            print(A)
            print("========================")
        # ========================================

        # Preparing alpha_i
        alpha_k = np.zeros(n_row)
        alpha_k[k+2:] = (A[k + 2:n_row,k].T/A[k + 1,k])

        # ========================================
        if verbose:
            print("alpha_{}".format(k+1))
            print(alpha_k)
            print("========================")
        # ========================================

        # Preparing e_i
        e_k = np.zeros(n_row)
        e_k[k+1] = 1

        # Calc. L_i
        M_k = np.eye(n_row) - np.outer(alpha_k, e_k)

        # ========================================
        if verbose:
            print("M_{}".format(k+1))
            print(M_k)
            print("========================")
        # ========================================

        # Transform system
        A = np.matmul(M_k, A)
        A = np.matmul(A, M_k.T)
        L = np.matmul(M_k, L)

        # ========================================
        if verbose:
            process_string = "M_{}.".format(k+1) + process_string +".M_^T{}".format(k+1)
            print(process_string)
            print(A)
            print("========================")
        # ========================================

    T = A
    L = np.linalg.inv(np.matmul(L, P.T))
    # ========================================
    if verbose:
        print("A final")
        print(A)
        print("========================")
    # ========================================
    return P, L, T

def Cond_number(A, norm, verbose=False):
	A_inv = Inverse(A, verbose)
	# ========================================
	if verbose:
	    print("A_^-1")
	    print(A_inv)
	    print("========================")
	# ========================================
	cond_number = np.linalg.norm(A, norm)*np.linalg.norm(A, norm)
	# ========================================
	if verbose:
	    print("Numero de condicion:")
	    print(cond_number)
	    print("========================")
	# ========================================
	return cond_number

def Cond_interval(A, x_tilde, b, norm, verbose=False):
    R = np.matmul(A, x_tilde) - b
    R_norm = np.linalg.norm(R, norm)
    # ========================================
    if verbose:
        print("R")
        print(R)
        print("========================")
        print("||R||")
        print(R_norm)
        print("========================")
    # ========================================
    b_norm = np.linalg.norm(b, norm)
    # ========================================
    if verbose:
        print("b")
        print(b)
        print("========================")
        print("||b||")
        print(b_norm)
        print("========================")
    # ========================================
    cond_A = Cond_number(A, norm, verbose)
    # ========================================
    if verbose:
        print("b")
        print(b)
        print("========================")
        print("||b||")
        print(b_norm)
        print("========================")
    # ========================================
    inf = R_norm/(b_norm*cond_A)
    sup = cond_A*R_norm/b_norm
    # ========================================
    if verbose:
        print("Cota de error")
        print("[{};{}]".format(inf, sup))
        print("========================")
    # ========================================
    return inf, sup

# Gram_Schmidt
#   A: Matrix in which we apply Gram Schmidt decomp
#   verbose: printing flag
#   pivot: Pivoting criteria
#
#   return: E, U such like A = EU
def Gram_Schmidt(A, verbose=False):
	n_row, n_col = A.shape[0], A.shape[1]
	E = np.zeros((n_row, n_col))
	U = np.zeros((n_col, n_col))

	n_row, n_col = A.shape[0], A.shape[1]

	for j in range(0,n_col):
		E[:, j] = A[:, j]

		if j > 1:
			for i in range(0, j-1):
				U[i,j] = E[:, i].dot(A[:, j])
				E[:, j] = E[:, j] -  U[i,j]*E[:, i]
		U[j,j] = np.sqrt(E[:, j].dot(E[:, j]))
		E[:, j] = E[:, j]/U[j,j]

		# ========================================
		if verbose:
			print("E en iter {}:".format(j))
			print(E)
			print("========================")
			print("U en iter {}:".format(j))
			print(U)
			print("========================")
	    # ========================================

	# ========================================
	if verbose:
		print("E final:")
		print(E)
		print("========================")
		print("U final:")
		print(U)
		print("========================")
	# ========================================
	return E, U


# HouseHolder
#	A: Matrix to wich Householder factorization is applied
#	return: Q, R Matrices so A = Q.R, Q:Orthogonal
def Householder(A, verbose=False):
	n_row, n_col = A.shape[0], A.shape[1]
	A_k = np.copy(A)
	n_row_k, n_col_k = A_k.shape[0], A_k.shape[1]
	R = np.zeros((n_row, n_col))
	Q = np.eye(n_row)

	# ========================================
	if verbose:
		print("=========================================")
		print("DESCOMPOSICION QR - HOUSEHOLDER")
		print("=========================================")
	# =========================================

	for k in range(0, n_col):
		# ========================================
		if verbose:
			print("-----------A_{}:-----------\n{}".format(k, A_k))
		# ========================================
		current_col = np.asmatrix(A_k[:, 0]).T

		e_k = np.asmatrix(np.repeat(0, A_k.shape[0])).T
		e_k[0,0] = 1

		w = np.sign(current_col[0,0])*np.linalg.norm(current_col, 2)*e_k + current_col
		if np.linalg.norm(w, 2) != 0:
			w = w/np.linalg.norm(w, 2)
		# ========================================
		if verbose:
			print("w_{}\n{}".format(k, w))
		# ========================================
		H_k = np.eye(A_k.shape[0]) - 2*np.matmul(w, w.T)
		# ========================================
		if verbose:
			print("H_{}\n{}".format(k, expandedMat(H_k, k)))
		# ========================================
		Q = np.matmul(expandedMat(H_k, k), Q)

		HxA = np.matmul(H_k, A_k)
		R[k:n_row,k:n_col] = np.copy(HxA)
		# ========================================
		if verbose:
			print("R_{}\n{}".format(k, R))
		# ========================================
		A_k = np.empty((n_row_k-1, n_col_k-1))
		A_k[:,:] = R[k+1:,k+1:]
		n_row_k -= 1
		n_col_k -= 1

	# ========================================
	if verbose:
		print("Q final:\n{}".format(Q.T))
		print("R final:\n{}".format(R))
	# ========================================
	return Q.T, R

def expandedMat(A, n):
	A_exp = np.copy(A)
	upper_expand = np.zeros((n,A.shape[1]))
	left_expand = np.vstack((np.eye(n), np.zeros((A.shape[0],n))))
	A_exp = np.r_[upper_expand, A_exp]
	A_exp = np.c_[left_expand, A_exp]
	return np.asmatrix(A_exp)


# Factorizacion A = QR mediante Matrices de rotacion de Givens
def Givens(A, verbose=False):
	# ========================================
	if verbose:
		print("=========================================")
		print("DESCOMPOSICION QR - GIVENS")
		print("=========================================")
	# =========================================

	new_A = np.copy(A)
	n = A.shape[0]
	Q = np.eye(n)

	for col in range(A.shape[1]):
		for row in range(A.shape[0] - 1, -1, -1):
			if row > col:
				# ====================================
				if verbose:
					print("=======================")
					print("({}, {})".format(row, col))
					print("=======================")
				# ====================================

				G_matrix = np.eye(n)
				a = new_A[row-1,col]
				b = new_A[row,col]

				cos = a/(np.sqrt( a**2 + b**2 ))
				sen = b/(np.sqrt( a**2 + b**2 ))

				# ====================================	
				if verbose:
					print("(cos, sen) = ({};{})".format(cos, sen))
				# ====================================

				G_matrix[row, row] = cos
				G_matrix[row, row-1] = sen
				G_matrix[row-1, row-1] = cos
				G_matrix[row-1, row] = -sen

				# ====================================
				if verbose:
					print("Matriz de Givens para (i,j)=({},{}):\n".format(row, col), G_matrix)
				# ====================================

				Q = np.matmul(G_matrix.T,Q)

				# ====================================
				if verbose:
					print("Matriz ingresada luego de la rotacion:\n",new_A)
				# ====================================

				new_A = np.matmul(G_matrix.T, new_A)
	R = np.copy(new_A)
	print("Matriz final: \n", R)
	print("Matriz Q (Producto de Matrices de Givens): \n", Q)
	return Q.T, R


# ===================================================
# Equation solving methods
# ===================================================

def solve_Gauss(A, b, verbose=False, pivot="none"):
	# ========================================
	if verbose:
		print("=========================================")
		print("METODO DE GAUSS")
		print("=========================================")
	#=========================================
	P, L, U = Gauss(A, b, verbose, pivot)
	# ========================================
	if verbose:
		print("De LUx = Pb, evaluamos Lz = Pb y Ux = z")
		print("1. Resolviendo Lz = Pb")
	# ========================================
	z = for_solve_triangular(L, np.matmul(P, b), verbose)
	# ========================================
	if verbose:
		print("z")
		print(z)
		print("2. Resolviendo Ux = z")
	# ========================================
	x = back_solve_triangular(U, z, verbose)
	# ========================================
	if verbose:
		print("x")
		print(x)
	# ========================================
	return x

def solve_Gauss_Jordan(A, b, verbose=False, pivot="none"):
	#=========================================
	if verbose:
		print("METODO DE GAUSS JORDAN")
	#=========================================
	return Gauss_Jordan(A, b, verbose, pivot)

def solve_LU_1(A, b, verbose=False, pivot="none"):
	#=========================================
	if verbose:
		print("METODO DE DESCOMPOSICION LU_1")
	#=========================================
	P, L, U = LU_1(A, verbose, pivot)
	# ========================================
	if verbose:
		print("De LUx = Pb, evaluamos Lz = Pb y Ux = z")
		print("1. Resolviendo Lz = Pb")
	# ========================================
	z = for_solve_triangular(L, np.matmul(P, b), verbose)
	# ========================================
	if verbose:
		print("z")
		print(z)
		print("2. Resolviendo Ux = z")
	# ========================================
	x = back_solve_triangular(U, z, verbose)
	# ========================================
	if verbose:
		print("x")
		print(x)
	# ========================================
	return x

def solve_L_1U(A, b, verbose=False, pivot="none"):
		#=========================================
		if verbose:
			print("=========================================")
			print("METODO DE DESCOMPOSICION L_1U")
			print("=========================================")
		#=========================================
		P, L, U = L_1U(A, verbose, pivot)
		# ========================================
		if verbose:
			print("De LUx = Pb, evaluamos Lz = Pb y Ux = z")
			print("1. Resolviendo Lz = Pb")
		# ========================================
		z = for_solve_triangular(L, np.matmul(P, b), verbose)
		# ========================================
		if verbose:
			print("z")
			print(z)
			print("2. Resolviendo Ux = z")
		# ========================================
		x = back_solve_triangular(U, z, verbose)
		# ========================================
		if verbose:
			print("x")
			print(x)
		# ========================================
		return x

def solve_LDLt(A, b, verbose=False, pivot="none"):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE DESCOMPOSICION LDL^t")
		print("=========================================")
		print("Partiendo de Ax = b se descompone:")
		print("* PA(P^t.P)x = Pb")
		print("'--> LDL^t.Px = Pb")
		print("Luego, la resolucion será: Lz = Pb -> Dv = z -> L^t.k = v -> Px = k")
	#=========================================
	P, L, D = LDLt(A, verbose, pivot)
	# ========================================
	if verbose:
		print("1. Resolviendo Lz = Pb")
	# ========================================
	z = for_solve_triangular(L, np.matmul(P, b), verbose)
	# ========================================
	if verbose:
		print("z")
		print(z)
		print("2. Resolviendo Dv = z")
	# ========================================
	v = for_solve_triangular(D, z, verbose)
	# ========================================
	if verbose:
		print("v")
		print(v)
		print("3. Resolviendo L^tk = v")
	# ========================================
	k = back_solve_triangular(L.T, v, verbose)
	# ========================================
	if verbose:
		print("k")
		print(k)
		print("4. Resolviendo Px = k (Ix = P^t.k)")
	# ========================================
	x = for_solve_triangular(np.eye(P.shape[0]), np.matmul(P.T, k), verbose)
	# ========================================
	if verbose:
		print("x")
		print(x)
	# ========================================
	return x

def solve_Cholesky(A, b, verbose=False, pivot="none"):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE DESCOMPOSICION DE CHOLESKY")
		print("=========================================")
		print("Partiendo de Ax = b se descompone:")
		print("* PA(P^t.P)x = Pb")
		print("'--> G.G^t.Px = Pb")
		print("Luego, la resolucion será: Gz = Pb -> G^t.v = z -> Px = v")
	#=========================================
	P, G = Cholesky(A, verbose, pivot)
	# ========================================
	if verbose:
		print("1. Resolviendo Gz = Pb")
	# ========================================
	z = for_solve_triangular(G, np.matmul(P, b), verbose)
	# ========================================
	if verbose:
		print("z")
		print(z)
		print("2. Resolviendo G^t.v = z")
	# ========================================
	v = back_solve_triangular(G.T, z, verbose)
	# ========================================
	if verbose:
		print("v")
		print(v)
		print("3. Resolviendo Px = v")
	# ========================================
	x = back_solve_triangular(np.eye(P.shape[0]), np.matmul(P.T, v), verbose)
	# ========================================
	if verbose:
		print("x")
		print(x)
	# ========================================
	return x

def solve_Parlett_Reid(A, b, verbose=False, pivot="none"):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE RESOLUCION DE PARLET & REID")
		print("=========================================")
		print("Partiendo de Ax = b se descompone:")
		print("* PA(P^t.P)x = Pb")
		print("'--> L.T.L^t.Px = Pb")
		print("Luego, la resolucion será: Lz = Pb -> T.v = z -> L^t.k = v -> Px = k")
	#=========================================
	P, L, T = Parlett_Reid(A, verbose, pivot)
	# ========================================
	if verbose:
		print("1. Resolviendo Lz = Pb")
	# ========================================
	z = for_solve_triangular(L, np.matmul(P, b), verbose)
	# ========================================
	if verbose:
		print("z")
		print(z)
		print("2. Resolviendo Tv = z")
	# ========================================
	print(T)
	v = solve_Gauss(T, z)
	#v  = solve_tridiagonal(T, z)
	#print("???????????????????????", v)
	# ========================================
	if verbose:
		print("v")
		print(v)
		print("3. Resolviendo L^t.k = v")
	# ========================================
	k = back_solve_triangular(L.T, v, verbose)
	# ========================================
	if verbose:
		print("k")
		print(k)
		print("4. Resolviendo Px = k (Ix = P^t.k)")
	# ========================================
	x = for_solve_triangular(np.eye(P.shape[0]), np.matmul(P.T, k), verbose)
	# ========================================
	if verbose:
		print("x")
		print(x)
	# ========================================
	return x

def solve_Gram_Schmidt(A, b, verbose=False, pivot="none"):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE RESOLUCION DE GRAM SCHMIDT")
		print("=========================================")
		print("Partiendo de Ax = b se descompone:")
		print("* A.x = b")
		print("'--> Ux = E^t.b")
	#=========================================
	E, U = Gram_Schmidt(A, verbose)
	x = back_solve_triangular(U, np.matmul(np.linalg.inv(E), b), verbose)
	# ========================================
	if verbose:
		print("x")
		print(x)
	# ========================================
	return x

def solve_Householder(A, b, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE RESOLUCION DE HOUSEHOLDER")
		print("=========================================")
	#=========================================

	if A.shape[1] <= A.shape[0]:
		#=========================================
		if verbose:
			print("Caso m(fila)>n(columnas):")
			print("Aplicado a A = Q.R")
			print("Entonces se partira de A.x = b")
			print("'---> Q.R.x = b")
			print("'---> 1) R.x = Q^t.b = b_hat")
			print("Usando solo la parte no nula de la matriz de R = [r  0]^T")
			print("'---> 2) r.x = mini_b_hat")
		#=========================================

		Q,R = Householder(A, verbose)

		b_hat = np.matmul(Q.T, b)
		mini_b_hat = b_hat[:, :R.shape[1]]

		# solucionando dimension de mini_b_hat
		temp = []
		for i in range(mini_b_hat.shape[1]):
			temp.append(mini_b_hat[0,i])
		mini_b_hat = np.array(temp)
		
		r = np.copy(R[:R.shape[1],:])

		#=========================================		
		if verbose:
			print("r:\n", r)
			print("mini_b_hat (b_hat util para matriz r):\n", mini_b_hat)
		#=========================================
		x = back_solve_triangular(r, mini_b_hat)
		#=========================================
		if verbose:
			print("finalmente x:\n{}".format(x))
		#=========================================
		return x
	
	else:
		#=========================================
		if verbose:
			print("Caso n(columnas)>m(filas):")
			print("Aplicado a A^T = Q.R => A = [R^T  0^T].Q^T")
			print("Entonces se partira de A.x = b")
			print("'---> [R^T  0^T].Q^Tx = b")
			print("'---> (Q^T)x = z & [R^T  0^T]z = b")
			print("'---> R^T . z_R = b_R & x = Q([z_R  0]^T)")
		#=========================================

		A_t = A.T
		Q,R = Householder(A_t, verbose)
		print("=========================================")

		R_t_0 = R.T
		#=========================================
		if verbose:
			print("[R^T 0^T]:\n{}".format(R_t_0))
		#=========================================
		R_t = R_t_0[:,:R_t_0.shape[0]]
		#=========================================
		if verbose:
			print("R^T:\n{}".format(np.round(R_t)))
		#=========================================
		b_R = b[:R_t.shape[0]]
		#=========================================
		if verbose:
			print("b_R:\n{}".format(b_R))
			print("=========================================")
			print("> Resolviendo R^T z_R = b_R")
			print("=========================================")
		#=========================================
		z_R = for_solve_triangular(R_t,b_R, verbose)
		#=========================================
		if verbose:
			print("z_R:\n{}".format(z_R))
		#=========================================
		#=========================================
		if verbose:
			print("b_R:\n{}".format(b_R))
			print("=========================================")
			print("> Resolviendo x = Q ([z_R  0]^T)")
			print("=========================================")
		#=========================================
		z_R_0 = np.append(z_R, np.zeros(Q.shape[0] - z_R.size))
		#=========================================
		if verbose:
			print("z_R:\n{}".format(z_R_0))
		#=========================================
		x = np.matmul(Q, z_R_0)
		#=========================================
		if verbose:
			print("finalmente x:\n{}".format(x))
		#=========================================
		return x

def solve_Jacobi(A, b, x_0, epsilon, n_iter=-1, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE RESOLUCION DE JACOBI")
		print("Se resolverá la ecuacion iterativa:")
		print("x = Jx + c, con J = I-D\'A y c = D\'b")
		print("=========================================")
	#=========================================

	D, E, F = decompose(A, verbose)

	#=========================================
	if verbose:
		print("========== Invirtiendo matriz D: =============")
	#=========================================
	D_inv = Inverse(D, verbose);

	J = np.eye(D_inv.shape[0]) - np.matmul(D_inv, A)
	#=========================================
	if verbose:
		print("J:\n{}".format(J))
	#=========================================

	x = x_0

	b_hat = np.matmul(D_inv, b)
	
	#=========================================
	if verbose:
		print("c:\n{}".format(b_hat))
	#=========================================

	if n_iter >= 0:
		for i in range(0, n_iter):
			if verbose:
				print("x_{}\n{}".format(i, x))
			if(np.linalg.norm(x - (np.matmul(J, x) + b_hat)) < epsilon):
				print("La solución converge con una variación de epsilon = {}\nSaliendo...".format(epsilon))
				break
			x = np.matmul(J, x) + b_hat
			
	return x

def solve_Gauss_Seidel(A, b, x_0, epsilon, n_iter=-1, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE RESOLUCION DE GAUSS-SEIDEL")
		print("Se resolverá la ecuacion iterativa:")
		print("x = Gx + c, con G = I-(D-E)\'A y c = (D-E)\'b")
		print("=========================================")
	#=========================================

	D, E, F = decompose(A, verbose)
	
	#=========================================
	if verbose:
		print("========== Invirtiendo matriz (D-E): =============")
	#=========================================

	DE_inv = Inverse(D - E)

	G = np.matmul(DE_inv, F)
	
	#=========================================
	if verbose:
		print("G:\n{}".format(G))
	#=========================================

	x = x_0

	b_hat = np.matmul(DE_inv, b)
	
	#=========================================
	if verbose:
		print("c:\n{}".format(b_hat))
	#=========================================

	if n_iter >=0 :
		for i in range(0, n_iter):
			if verbose:
				print("x_{}\n{}".format(i, x))
			if(np.linalg.norm(x - (np.matmul(G, x) + b_hat)) < epsilon):
				print("La solución converge con una variación de epsilon = {}\nSaliendo...".format(epsilon))
				break
			x = np.matmul(G, x) + b_hat	

	return x

def solve_SOR(A, b, x_0, w, epsilon, n_iter=-1, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE RESOLUCION SOR")
		print("Se resolverá la ecuacion iterativa:")
		print("x = Sx + c, con S = (D-wE)\'[(1-w)D + wF] y c = (D-wE)\'b")
		print("=========================================")
	#=========================================

	D, E, F = decompose(A, verbose)
	
	D_wE = D - w*E

	#=========================================
	if verbose:
		print("(D - wE):\n{}".format(D_wE))		
		print("========== Invirtiendo matriz (D - wE): =============")
	#=========================================
	D_wE_inv = Inverse(D_wE, verbose);

	S = np.matmul( D_wE_inv, ( ((1-w)*D) + (w*F) ) )
	#=========================================
	if verbose:
		print("S:\n{}".format(S))
	#=========================================

	x = x_0

	b_hat = w*np.matmul(D_wE_inv, b)
	
	#=========================================
	if verbose:
		print("c:\n{}".format(b_hat))
	#=========================================

	if n_iter >= 0:
		for i in range(0, n_iter):
			if verbose:
				print("x_{}\n{}".format(i, x))
			if(np.linalg.norm(x - (np.matmul(S, x) + b_hat)) < epsilon):
				print("La solución converge con una variación de epsilon = {}\nSaliendo...".format(epsilon))
				break
			x = np.matmul(S, x) + b_hat
			
	return x

def solve_SSOR(A, b, x_0, w, epsilon, n_iter=-1, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE RESOLUCION SSOR")
		print("Se resolverá la ecuacion iterativa:")
		print("x = SSx + c, con:")
		print("-> SS = (D-wF)\'[(1-w)D + wE](D-wE)\'[(1-w)D + wF]")
		print("-> c = w(D-wF)\'{ [(1-w)D + wE](D-wE)\' + I }\'b")
		print("=========================================")
	#=========================================

	D, E, F = decompose(A, verbose)
	
	D_wE = D - w*E
	D_wF = D - w*F

	#=========================================
	if verbose:
		print("(D - wE):\n{}".format(D_wE))
		print("(D - wF):\n{}".format(D_wF))		
		print("========== Invirtiendo matriz (D - wE): =============")
	#=========================================
	D_wE_inv = Inverse(D_wE, verbose)

	#=========================================
	if verbose:
		print("========== Invirtiendo matriz (D - wF): =============")
	#=========================================
	
	D_wF_inv = Inverse(D_wF, verbose)

	SS_1 = np.matmul( D_wF_inv, ( ((1-w)*D) + (w*E) ) )
	SS_2 = np.matmul( D_wE_inv, ( ((1-w)*D) + (w*F) ) )
	SS = np.matmul(SS_1, SS_2)
	
	#=========================================
	if verbose:
		print("SS:\n{}".format(SS))
	#=========================================

	x = x_0

	b_hat = np.matmul( (1-w)*D + w*E , D_wE_inv) + np.eye(D.shape[0])
	b_hat = w*np.matmul(D_wF_inv, b_hat)
	b_hat = np.matmul(b_hat, b)
	
	#=========================================
	if verbose:
		print("c:\n{}".format(b_hat))
	#=========================================

	if n_iter >= 0:
		for i in range(0, n_iter):
			if verbose:
				print("x_{}\n{}".format(i, x))
			if(np.linalg.norm(x - (np.matmul(SS, x) + b_hat)) < epsilon):
				print("La solución converge con una variación de epsilon = {}\nSaliendo...".format(epsilon))
				break
			x = np.matmul(SS, x) + b_hat
			
	return x

def solve_CGradient(A, b, x_0, n_iter, epsilon, delta, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE RESOLUCION POR GRADIENTE CONJUGADO")
		print("=========================================")
	#=========================================

	r = b - np.matmul(A, x_0)
	v = r[:]
	c = r.dot(r)
	x = x_0

	print(r)
	print(v)
	print(c)

	for i in range(1, n_iter):
		if np.sqrt(v.dot(v)) < delta:
			break
		z = np.matmul(A, v)
		t = c/v.dot(z)
		x = x + t*v
		r = r - t*z
		d = r.dot(r)
		if d < epsilon:
			print("La solución converge con una variación de epsilon = {}\nSaliendo...".format(epsilon))
			break
		v = r + (d/c)*v
		c = d
		print("x_{}\n{}".format(i, x))
	return x

# Non-Linear (NL) equations solving methods:

def NL_solve_Newton(x_0, f, df, max_iter, E, verbose=False):
	#=========================================
	if verbose:
		print("===========================================")
		print("METODO DE RESOLUCION DE NEWTON: EC. NO LIN.")
		print("===========================================")
	#=========================================
	x_old = np.copy(x_0)
	x_new = np.copy(x_0)

	# ===========================================================================
	if verbose:
		print("=========================================================================================")
		print("i\t\t\tx\t\t\tf(x)\t\t\tf\'(x)")
		print("=========================================================================================")
	# ===========================================================================
	for i in range(0, max_iter):
		# ===========================================================================
		if verbose:
			print("{}\t\t\t{}\t\t\t{}\t\t\t{}".format(i, x_new, f(x_new), df(x_new)))
		# ===========================================================================
		x_new = x_old - f(x_old)/df(x_old)
		if np.abs(x_new-x_old)/np.abs(x_new) < E:
			# ===========================================================================
			if verbose:
				print("Converge en iter:{}\nResultado:{}".format(i, x_new))
			# ===========================================================================
			return x_new
		x_old = np.copy(x_new)
	# ===========================================================================
	if verbose:
		print("Max. iteraciones alcanzado.\nResultado:{}".format(x_new))
	# ===========================================================================
	return x_new

def NL_solve_NewtonMod(x_0, f, df, ddf, max_iter, E, verbose=False):
	#=========================================
	if verbose:
		print("=================================================")
		print("METODO DE RESOLUCION DE NEWTON MODIF: EC. NO LIN.")
		print("=================================================")
	#=========================================
	x_old = np.copy(x_0)
	x_new = np.copy(x_0)

	# ===========================================================================
	if verbose:
		print("=========================================================================================")
		print("i\t\t\tx\t\t\tf(x)\t\t\tf\'(x)\t\t\tf\'\'(x)")
		print("=========================================================================================")
	# ===========================================================================
	for i in range(0, max_iter):
		# ===========================================================================
		if verbose:
			print("{}\t\t\t{}\t\t\t{}\t\t\t{}\t\t\t{}".format(i, x_new, f(x_new), df(x_new), ddf(x_new)))
		# ===========================================================================
		x_new = x_old - (f(x_old)*df(x_old))/(df(x_old)**2 - f(x_old)*ddf(x_old))
		if np.abs(x_new-x_old)/np.abs(x_new) < E:
			# ===========================================================================
			if verbose:
				print("Converge en iter:{}\nResultado:{}".format(i, x_new))
			# ===========================================================================
			return x_new
		x_old = np.copy(x_new)
	# ===========================================================================
	if verbose:
		print("Max. iteraciones alcanzado.\nResultado:{}".format(x_new))
	# ===========================================================================
	return x_new

def NL_solve_Secante(x_0, f, max_iter, E, verbose=False):
	#=========================================
	if verbose:
		print("===========================================")
		print("METODO DE RESOLUCION DE SECANTE: EC. NO LIN.")
		print("===========================================")
	#=========================================
	x_2old = np.copy(x_0)*1.01 # Jugar con este valor
	x_old = np.copy(x_0)
	x_new = np.copy(x_0)

	# ===========================================================================
	if verbose:
		print("======================================================")
		print("i\t\t\tx\t\t\tf(x)")
		print("======================================================")
	# ===========================================================================
	for i in range(0, max_iter):
		# ===========================================================================
		if verbose:
			print("{}\t\t\t{}\t\t\t{}".format(i, x_new, f(x_new)))
		# ===========================================================================
		x_new = x_old - (x_old-x_2old)*f(x_old)/(f(x_old)-f(x_2old))
		if np.abs(x_new-x_old) < E:
			# ===========================================================================
			if verbose:
				print("Converge en iter:{}\nResultado:{}".format(i, x_new))
			# ===========================================================================
			return x_new
		x_2old = np.copy(x_old)
		x_old = np.copy(x_new)
	# ===========================================================================
	if verbose:
		print("Max. iteraciones alcanzado.\nResultado:{}".format(x_new))
	# ===========================================================================
	return x_new

def NL_solve_ReguleFalsi(a_0, b_0, E, f, verbose=False):
	#=========================================
	if verbose:
		print("===========================================")
		print("METODO DE RESOLUCION DE REGULE FALSI: EC. NO LIN.")
		print("===========================================")
	#=========================================
	a = a_0
	b = b_0
	c = (a*f(b) - b*f(a))/(f(b) - f(a))
	i = 0
	# ===========================================================================
	if verbose:
		print("======================================================")
		print("i\t\t\ta\t\t\tc\t\t\tb\t\t\tf(c)")
		print("======================================================")
	# ===========================================================================
	while np.abs(f(c)) > E:
		# ===========================================================================
		if verbose:
			print("{}\t\t\t{}\t\t\t{}\t\t\t{}\t\t\t{}".format(i, a, c, b, f(c)))
		# ===========================================================================
		c = (a*f(b) - b*f(a))/(f(b) - f(a))
		if f(c) == 0:
			a = c
			b = c
			break
		if np.sign(f(a)) == np.sign(f(c)):
			b = c
		else:
			a = c
		i += 1
	return c

def NL_solve_Biseccion(a_0, b_0, E, f, verbose=False):
	#=========================================
	if verbose:
		print("===========================================")
		print("METODO DE RESOLUCION POR BISECCION: EC. NO LIN.")
		print("===========================================")
	#=========================================
	a = a_0
	b = b_0
	c = (a*f(b) - b*f(a))/(f(b) - f(a))
	i = 0

	# ===========================================================================
	if verbose:
		print("======================================================")
		print("i\t\t\ta\t\t\tc\t\t\tb\t\t\tf(c)")
		print("======================================================")
	# ===========================================================================
	while (b - a)/2 > E:
		# ===========================================================================
		if verbose:
			print("{}\t\t\t{}\t\t\t{}\t\t\t{}\t\t\t{}".format(i, a, c, b, f(c)))
		# ===========================================================================
		c = (a+b)/2
		print("iter:{}\ta: {}\t\tc: {}\t\tb: {}\t\t(b - a)/2: {}".format(i, np.round(a, 10), np.round(c, 10), np.round(b, 10), (b - a)/2))

		if f(c) == 0:
			a = c
			b = c
			break

		if np.sign(f(a)) == np.sign(f(c)):
			a = c
		else:
			b = c

		i += 1
	return np.round(c, 10)

# NL system solving methods
# Recordar poner antes de usar estos metodos:
"""
#=========================================
	if verbose:
		print("===========================================")
		print("METODO DE RESOLUCION DE [?]: SIST. NO LIN.")
		print("===========================================")
#=========================================
"""

def NLS_solve_Newton(x_0, f, J, max_iter, E, verbose=False):
	x_old = np.copy(x_0)
	x_new = np.copy(x_0)
	for i in range(0, max_iter):
		J_inv = Inverse(J(x_old))
		x_new = x_old - np.matmul(J_inv, f(x_old))
		if np.linalg.norm(x_new-x_old)/np.linalg.norm(x_new) < E:
			# ===========================================================================
			if verbose:
				print("Converge en iter:{}\nResultado:\n{}".format(i, x_new))
			# ===========================================================================
			return x_new
		x_old = np.copy(x_new)
	# ===========================================================================
	if verbose:
		print("Max. iteraciones alcanzado.\nResultado:\n{}".format(x_new))
	# ===========================================================================
	return x_new


# Calcular D: Tomar J -> D
def NLS_solve_Jacobi(x_0, f, D, max_iter, E, verbose=False):
	return NLS_solve_Newton(x_0, f, D, max_iter, E, verbose)

# Calcular L: Tomar J -> L
def NLS_solve_Gauss_Seidel(x_0, f, L, max_iter, E, verbose=False):
	return NLS_solve_Newton(x_0, f, L, max_iter, E, verbose)

# Calcular SOR_mat:
# Tomar J ->  L y D
# Tomar w
# SOR_mat = [ ( (1-w)/w ) D + L ]
def NLS_solve_SOR(x_0, f, SOR_mat, max_iter, E, verbose=False):
	return NLS_solve_Newton(x_0, f, SOR_mat, max_iter, E, verbose)


# Eigenvalues calculation methods:
def EIG_Krylov(A, y, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE KRYLOV")
		print("Resolviendo a partir de un vector inicial:", y)
		print("Para la matriz:\n", A)
		print("=========================================")
	#=========================================

	vec_list = []
	vec_list.append(y)

	n_iter = A.shape[0]
	for i in range(0, n_iter):
		#=========================================
		if verbose:
			print("=========================================")
			print("v_{} = A.v_{}: \n".format(i, i-1), np.matmul(A, vec_list[i]))
		#=========================================
		vec_list.append(np.matmul(A, vec_list[i]))

	coef_mat = np.copy(vec_list[0])

	for i in range(1, len(vec_list) - 1):
		coef_mat = np.c_[vec_list[i], coef_mat]
	b = -1.*vec_list[len(vec_list) - 1]

	#=========================================
	if verbose:
		print("Matriz de coef:\n", coef_mat)
		print("Vector de resolucion:\n", b)
		print("Resolviendo ...")
	#=========================================

	equation_coefs = solve_Gauss(coef_mat, b, verbose)

	#=========================================
	if verbose:
		eq_string = "lambda^" + str(n_iter)
		for i in range(0, n_iter):
			eq_string += "+(" + str(equation_coefs[i]) + ")lambda^" + str(n_iter-i-1) + " "
		eq_string += "= 0"
		print("=========================================")
		print("Ecuacion final:", eq_string)
		print("=========================================")
	#=========================================
	return equation_coefs

def EIG_Leverrier(A, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE LEVERRIER FADEEV")
		print("=========================================")
	#=========================================
	n = A.shape[0]
	B0 = np.copy(A)
	b_0 = -np.trace(B0)

	b_values = np.empty(n)
	b_values[0] = b_0

	B_old = B0
	#=========================================
	if verbose:
		print("---------------------------------------")
		print("B_{}:\n{}".format(1, B_old))
		print("b_{}:{}".format(1, b_values[0]))
		print("---------------------------------------")
	#=========================================
	for i in range(1, n):
		Bk = np.matmul(A, B_old + b_values[i-1]*np.eye(n))
		b_values[i] = -np.trace(Bk)/(i+1)
		B_old = np.copy(Bk)
		#=========================================
		if verbose:
			print("---------------------------------------")
			print("B_{}:\n{}".format(i+1, B_old))
			print("b_{}:{}".format(i+1, b_values[i]))
			print("---------------------------------------")
		#=========================================

	#=========================================
	if verbose:
		eq_string = "lambda^" + str(n)
		for i in range(0, n):
			eq_string += "+(" + str(b_values[i]) + ")lambda^" + str(n-i-1) + " "
		eq_string += "= 0"
		print("=========================================")
		print("Ecuacion final:", eq_string)
		print("=========================================")
	#=========================================	
	return b_values


def EIG_Potencia(A, x_0, epsilon, n_iter, norm, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE POTENCIA")
		print("=========================================")
	#=========================================
	x_k = x_0
	lambda_k = 0
	lambda_k_old = -1e100

	for i in range(0, n_iter):
		z_k = np.matmul(A, x_k)
		x_k = z_k/np.linalg.norm(z_k, norm)
		lambda_k = np.matmul(x_k.T, z_k)
		
		# ==================================================
		if verbose:
			print("=== i:{} ===\nx_k:\n{}\nlamdba:\n{}\n".format(i, x_k, lambda_k))
		# ==================================================
		if np.abs(lambda_k - lambda_k_old) < epsilon:
			break;

		lambda_k_old = lambda_k
	return lambda_k, x_k

def EIG_Potencia_Inversa(A, x_0, epsilon, n_iter, norm, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE POTENCIA INVERSA")
		print("=========================================")
	#=========================================
	x_k = np.copy(x_0)
	mu_k = 0
	mu_k_old = -1e100

	for i in range(0, n_iter):
		z_k = np.linalg.solve(A, x_k)
		x_k = z_k/np.linalg.norm(z_k, norm)
		mu_k = np.matmul(x_k.T, z_k)
		
		# ==================================================
		if verbose:
			print("=== i:{} ===\nx_k:\n{}\nmu:\n{}\n".format(i, x_k, mu_k))
		# ==================================================
		if np.abs(mu_k - mu_k_old) < epsilon:
			break;

		mu_k_old = mu_k
	return (1.0/mu_k), x_k

def EIG_Potencia_Inversa_Desp(A, x_0, lambda_bar, epsilon, n_iter, norm, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE POTENCIA INVERSA DESPLAZADA")
		print("=========================================")
	#=========================================
	x_k = np.copy(x_0)
	mu_k = 0
	mu_k_old = -1e100
	C = A - lambda_bar*np.eye(A.shape[0])
	
	for i in range(0, n_iter):		
		z_k = np.linalg.solve(C, x_k)
		x_k = z_k/np.linalg.norm(z_k, norm)
		mu_k = np.matmul(np.matmul(x_k.T, A), x_k)
		
		# ==================================================
		if verbose:
			print("=== i:{} ===\nx_k:\n{}\nmu:\n{}\n".format(i, x_k, mu_k))
		# ==================================================
		if np.abs(mu_k - mu_k_old) < epsilon:
			break;

		mu_k_old = mu_k
	return (lambda_bar +  1/mu_k), x_k

# Bisec_Pk
    # A: Tridiagonal matrix
	# k: lambda polynomial index

    # return: lambda polynomial for bisection eigenvalues finding method 
def Bisec_Pk(A, k):
	a = np.diag(A)
	b = np.diag(A, k=-1)

	if k == 0:
		def P0(lamb):
			return 1
		return P0
	elif k == 1:
		def P1(lamb):
			return (a[0] - lamb) 
		return P1
	else:
		f_k1 = Bisec_Pk(A, k-1)
		f_k2 = Bisec_Pk(A, k-2)

		def Pk(lamb):
			return ((a[k-1] - lamb)*f_k1(lamb) - (b[k-2]**2)*f_k2(lamb))
		return Pk

# Bisec_change_counter
	# v: array/vector of numeric values

    # return: number of times that the elements of v changed sign 
def Bisec_change_counter(v):
	cont = 0;
	for i in range(1, len(v)):
		sign_vi = 0 if (v[i] < 0) else 1
		sign_vi_1 = 0 if (v[i-1] < 0) else 1
		if sign_vi != sign_vi_1:
			cont += 1
	return cont


# Bisec_find_interval
	# A: Tridiagonal matrix
	# alpha_left: initial left point of search of interval
	# alpha_right: initial right point of search of interval
    
	# return: new interval [alpha_left, alpha_right] that satisfies condition 
def Bisec_find_interval(A, alpha_left, alpha_right):
	n = A.shape[0]
	# Creating {P0, P1, ..., Pn} redy to evaluate in a alpha
	p = []
	for i in range(0, n+1):
		p.append(Bisec_Pk(A, i))

	# Initial value of {P0(alpha_left), P1(alpha_left), ...}
	for_alpha_left = []
	for i in range(0, len(p)): for_alpha_left.append(p[i](alpha_left))

	# Initial value of {P0(alpha_right), P1(alpha_right), ...}
	for_alpha_right = []
	for i in range(0, len(p)): for_alpha_right.append(p[i](alpha_right))

	while (Bisec_change_counter(for_alpha_left) != 0) or (Bisec_change_counter(for_alpha_right) != n):
		# Extend the interval
		alpha_left -= 1
		alpha_right += 1

		for_alpha_left = []
		for i in range(0, len(p)): for_alpha_left.append(p[i](alpha_left))
		for_alpha_right = []
		for i in range(0, len(p)): for_alpha_right.append(p[i](alpha_right))


	print(alpha_left, alpha_right, '\n', for_alpha_left, for_alpha_right)
	return alpha_left, alpha_right

def Bisec_p_vector(A, alpha):
	n = A.shape[0]
	# Creating {P0, P1, ..., Pn} redy to evaluate in a alpha
	p = []
	for i in range(0, n+1):
		p.append(Bisec_Pk(A, i))

	for_alpha = []
	for i in range(0, len(p)): for_alpha.append(p[i](alpha))

	return np.array(for_alpha)

eigenvalues_container = [] # Global array for the recursion in Bisec_find_eigens

def Bisec_find_eigens(A, alpha_left, alpha_right, eps, num_eig):
	global eigenvalues_container
	print("[{};{}]".format(alpha_left, alpha_right))
	alpha_mid = (alpha_left + alpha_right)/2.0
	print("m=", alpha_mid)
	if np.abs(alpha_left - alpha_right) <= eps:
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!", alpha_mid)
		eigenvalues_container.append(alpha_mid)
	else:
		#print("for left:", Bisec_p_vector(A, alpha_mid))
		num_eig_left = Bisec_change_counter(Bisec_p_vector(A, alpha_mid))
		num_eig_right = num_eig - num_eig_left

		print("(#",num_eig_left, ";#",num_eig_right, ")")

		if num_eig_left > 0:
			Bisec_find_eigens(A, alpha_left, alpha_mid, eps, num_eig_left)
		if num_eig_right > 0:
			Bisec_find_eigens(A, alpha_mid, alpha_right, eps, num_eig_right)

def EIG_Biseccion(A, alpha_left_0, alpha_right_0, eps, verbose=False):
	global eigenvalues_container
	eigenvalues_container = []
	n = A.shape[0]
	alpha_left, alpha_right = Bisec_find_interval(A, alpha_left_0, alpha_right_0)
	Bisec_find_eigens(A, alpha_left, alpha_right, eps, n)
	return eigenvalues_container

def EIG_QR(A, max_iter, epsilon, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO QR")
		print("=========================================")
	#=========================================
	cont = 1
	A_k = np.copy(A)
	v_old = np.diag(A_k)
	#=========================================
	if verbose:
		print("A_0:\n{}".format(A_k))
	#=========================================
	for k in range(1, max_iter):
		cont += 1
		Q_k, R_k = Householder(A_k, verbose)
		A_k = np.matmul(R_k, Q_k)
		#=========================================
		if verbose:
			print("A_{}:\n{}".format(k, A_k))
		#=========================================
		if 	np.linalg.norm(A_k - np.diag(np.diag(A_k))) < epsilon:
			print("Valores hallados con presicion eps:{}".format(epsilon))
			break
		v_old = np.copy(np.diag(A_k))

	#=========================================
	if verbose:
		print("A_k final (iteracion {}):\n{}".format(cont, A_k))
		print("Valores propios:\n{}".format(np.diag(A_k)))
	#=========================================

	return np.diag(A_k)

def EIG_Jacobi(matrix, E, verbose=False, pre_check=True):
	if pre_check:
		# checking for squareness
		if matrix.shape[0] != matrix.shape[1]:
			raise (ValueError('Error: Matriz no es cuadrada'))

		# checking for symmetry
		for i in range(len(matrix)):
			for j in range(len(matrix)):
				if matrix[i][j] != matrix[j][i]:
					raise (ValueError('Error: Matriz no es simetrica'))

	counter = 0

	while 1:
		maximum, h, f = 0, 0, 0

		for i in range(len(matrix)):
			for j in range(len(matrix[i])):
				if i != j and maximum < abs(matrix[i][j]):
					maximum = abs(matrix[i][j])
					h, f = i, j

		m = (matrix[f][f] - matrix[h][h]) / 2 * matrix[h][f]
		t = np.sign(m) / (abs(m) + np.sqrt(m * m + 1))
		c = 1 / np.sqrt(t * t + 1)
		s = t * c

		mul_matrix = np.eye(len(matrix))
		mul_matrix[h][h], mul_matrix[f][f] = c, c
		mul_matrix[h][f], mul_matrix[f][h] = s, -s
		matrix = np.dot(np.dot(mul_matrix.transpose(), matrix), mul_matrix)

		counter += 1

		if verbose:
			s_d = sum(matrix[i][i] * matrix[i][i] for i in range(len(matrix)))
			s_nd = sum(matrix[i][j] * matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix[i])) if i != j)

			print('iter#', counter,  '\n\nMatriz T:\n', mul_matrix, '\n\nMatriz T\':\n', mul_matrix.transpose(), '\n\n',
				  'S_d: ', s_d, '\n', 'S_nd: ', s_nd, '\n', 'S: ', s_d + s_nd, '\n---------------------------------')

		if sum(matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix[i])) if i != j) < E:
			break

	return matrix.diagonal()[::-1]

# Interpolation methods: 

# Ln_i: Retorna el coeficiente de Lagrange L_ni
def Ln_i(n, i, x_vector, x):
	p1 = 1
	p2 = 1

	for j in range(0, n):
		if i != j:
			p1 *= (x - x_vector[j])
			p2 *= (x_vector[i] - x_vector[j])
	
	return p1/p2

# INT_Lagrange: Retorna el polinomio generado P_n_eq, listo para evaluar cualquier
# x y tomar el valor de la curva interpolada en cualquier punto
def INT_Lagrange(x_vector, y_vector, verbose=False):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE LAGRANGE")
		print("Este metodo genera una funcion P_n(x)\nrepresentando al polinomio interpolado por Lagrange")
		print("Se usara los datos de \'x\':", x_vector)
		print("Se usara los datos de \'y\':", y_vector)
		print("=========================================")
	#=========================================

	def P_n_eq(x, verbose=False):
		#=========================================
		if verbose:
			print("-----------------------------------")
			print("Evaluando x = {}".format(x))
		#=========================================
		n = x_vector.size
		acc = 0
		for i in range(y_vector.size):
			#=========================================
			if verbose:
				print("L_{},{}:{}".format(n, i, Ln_i(n, i, x_vector, x)))
			#=========================================
			acc += y_vector[i]*Ln_i(n, i, x_vector, x)

		#=========================================
		if verbose:
			print("Finalmente, P_{}({}):{}".format(n, x, acc))
			print("-----------------------------------")
		#=========================================
		return acc
	return P_n_eq

# INT_Lagrange_error: Retorna el polinomio de error generado P_n_eq,
# listo para evaluar cualquier c dentro del intervalo [a,b] en el
# que pertenecen los valores de x_vector
def INT_Lagrange_error(x_vector, dfn_plus_1, c, verbose=True):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE ERROR DE LAGRANGE")
		print("Este metodo genera una funcion E_n(x)\nrepresentando al polinomio de error de la funcion interpolada por Lagrange")
		print("Se usara los datos de \'x\':", x_vector)
		print("Se usara como c en el interv. [\'x\']:", c)
		print("=========================================")
	#=========================================
	if (c < x_vector[0]) or (c > x_vector[x_vector.size-1]):
			print("Advertencia: El valor de c ingresado no esta dentro de los limites de x")

	def E_n(x):
		#=========================================
		if verbose:
			print("-----------------------------------")
			print("Evaluando x = {}".format(x))
		#=========================================

		acc1 = 1 # product of (x-x_i)'s
		acc2 = 1 # (n+1)!
		n = x_vector.size

		for i in range(0,n):
			acc1 *= (x - x_vector[i])
			acc2 *= i+1
		
		result = (acc1 * dfn_plus_1(c))/acc2
		#=========================================
		if verbose:
			print("Finalmente, (x-x_0)...(x-x_{})*f^({})({}) / {}! = {}".format(n-1, n, c, n,result))
		#=========================================		
		return result
	return E_n

def divdif(f, x_vector):
	n = x_vector.size
	if n is 1:
		return( f(x_vector) )
	else:
		x_copy_1 = np.copy(x_vector[1:])
		x_copy_2 = np.copy(x_vector[:n-1])
		return ( divdif(f, x_copy_1) - divdif(f, x_copy_2) )/(x_vector[x_vector.size-1] - x_vector[0])

def Newton_interpol_table(f, x_vector, verbose=False):
	n = x_vector.size
	table = np.empty((n, n)) # Tabla a generar
	table_aux = np.empty((n, 2))


	for i in range(table_aux.shape[0]):
		table_aux[i, 0] = i
		table_aux[i, 1] = x_vector[i]

	for i in range(table.shape[0]):
		for j in range(table.shape[1]):
			if (j <= i):
				table[i, j] = divdif(f, x_vector[i-j: i+1])
			else:
				table[i, j] = np.nan

	if verbose:
		print("Tabla de coeficientes via diferencia dividida:")
		table_head = "k\tx_k\t"
		for i in range(table.shape[1]):
			table_head += "f[x_k-{} -> x_k] | ".format(i)
		print(table_head)
		for i in range(table.shape[0]):
			print(table_aux[i, 0], '|', table_aux[i, 1], '|', end='')
			for j in range(table.shape[1]):
				print(table[i, j], "|\t", end='')
				if np.isnan(table[i, j]):
					print("\t\t", end='')
			print()

	return table

def INT_Newton(f, x_vector, verbose=False, my_coefs=None):
	#=========================================
	if verbose:
		print("=========================================")
		print("METODO DE NEWTON")
		print("Este metodo genera una funcion P_n(x)\nrepresentando al polinomio interpolado por Newton")
		print("Se usara los datos de \'x\':", x_vector)
		print("=========================================")
	#=========================================
	
	if my_coefs is None:
		#=========================================
		if verbose:
			print("Coeficientes no proporcionados.")
			print("Se generaran coefs. via diferencias divididas.")
		#=========================================
		coefs = Newton_interpol_table(f, x_vector, verbose)
		coefs = np.diag(coefs)
	
	else:
		coefs = np.copy(my_coefs)

	#=========================================
	if verbose:
		print("Los coeficientes de la curva interpolada seran: ")
		print(coefs)
	#=========================================

	str_eq = ""
	str_eq += str(coefs[coefs.size - 1])

	for i in range(coefs.size-1):
		str_eq = "(x - {}).[".format(x_vector[x_vector.size - i - 2]) + str_eq
		str_eq = "{} + ".format(coefs[coefs.size - i - 2]) + str_eq
	for i in range(coefs.size-1):
		str_eq += ']'
	str_eq = "P_{}(x) = ".format(coefs.size - 1) + str_eq


	def P_n_eq(x):
		acc = coefs[coefs.size - 1]

		for i in range(coefs.size-1):
			acc *= (x - x_vector[x_vector.size - i - 2])
			acc += coefs[coefs.size - i - 2]

		return acc
	#=========================================
	if verbose:
		print("---------------------------------------------------------")
		print("Polinomio de interpolacion generado:\n{}".format(str_eq))
		print("---------------------------------------------------------")
	#=========================================	
	return P_n_eq

