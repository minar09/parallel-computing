import numpy as np
from scipy.linalg import solve

def jacobi(A, b, x, n):

    D = np.diag(A)
    R = A - np.diagflat(D)
    
    for i in range(n):
        x = (b - np.dot(R,x))/ D
        print(str(i).zfill(3)),
        print(x)
    return x

'''___Main___'''

A = np.array([[4.0, -2.0, 1.0], [1.0, -3.0, 2.0], [-1.0, 2.0, 6.0]])
b = [1.0, 2.0, 3.0]
x = [1.0, 1.0, 1.0]
n = 25

print("\n\ninit"),
print(x)
print("")
x = jacobi(A, b, x, n)
print("\nSol "),
print(x)
print("Act "),
print(solve(A, b))
print("\n")
