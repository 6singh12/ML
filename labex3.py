import pandas as pd
import numpy as np

# Load the Excel file into a DataFrame
df = pd.read_excel('Lab Session1 Data.xlsx', sheet_name='Purchase data')

# Extracting the relevant columns
A = df.iloc[:, 1:-1].values.astype(float)  # Convert to numerical values
C = df.iloc[:, -1].values.astype(float)    # Convert to numerical values

# Dimensionality of the vector space
dimensionality = A.shape[1]

# Number of vectors in the vector space
num_vectors = A.shape[0]

# Rank of Matrix A
rank_A = np.linalg.matrix_rank(A)

# Using Pseudo-Inverse to find the cost of each product
pseudo_inverse_A = np.linalg.pinv(A)
cost_per_product = np.dot(pseudo_inverse_A, C)

# Print results
print("Dimensionality of the vector space:", dimensionality)
print("Number of vectors in the vector space:", num_vectors)
print("Rank of Matrix A:", rank_A)
print("Cost of each product available for sale:")
for i, cost in enumerate(cost_per_product):
    print(f"Product {i+1}: {cost}")
