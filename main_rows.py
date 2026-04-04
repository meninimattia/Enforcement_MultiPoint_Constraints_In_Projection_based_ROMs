import numpy as np

def find_main_rows(matrix):
	vect = [] # Vector of the "main" rows (they correspond to the main DoFs -> in the 1D case DoFs == Nodes)
	columns = []
	mask = np.zeros(matrix.shape[1]) # To control if a particular column of T is already controlled
	# This is done in order to keep only the first 0...0i0...0 type of rows
	row = 0 # To control all the rows of the matrix T
	row_length = matrix.indptr[1] # Initilization of the length of each row (== number of values in that particular row)
	for idx, col in enumerate(matrix.indices): # for cycle over the columns 
		if idx == matrix.indptr[row+1]: # used to change the row (the indptr points to the first value of each row)
			row += 1 # row controlled 
			row_length = matrix.indptr[row+1]-matrix.indptr[row] # Number of entries in the particular row
		# print(f"row {row}, col {col}, val {matrix.data[idx]}")
		if not(mask[col]) and row_length==1: # Storage only of the rows related to the columns never seen and with only one value inside
			vect.append(row) # indices of the row stored
			columns.append(int(col))
			mask[col] = 1.0 # column seen
	order = np.zeros(len(columns), dtype=int)
	for i in range(0, len(columns)):
		order[columns[i]] = i
	return vect, order