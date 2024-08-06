import numpy as np
from .cka import CKA

def test_cka():
    # Create a random input
    X = np.random.rand(4, 64)
    # CKA of X with itself should be 1
    cka_identical = CKA(X, X)
    print(f"CKA of identical inputs: {cka_identical}")
    assert np.isclose(cka_identical, 1.0), "CKA of identical inputs should be 1"

    # Create orthogonal matrices
    X = np.random.rand(100, 64)
    Y = np.random.rand(100, 64)
    cka_orthogonal = CKA(X, Y)
    print(f"CKA of orthogonal inputs: {cka_orthogonal}")
    assert cka_orthogonal < 0.1, "CKA of orthogonal inputs should be close to 0"

    X = np.random.rand(100, 64)
    Y = 2 * X
    cka_scaled = CKA(X, Y)
    print(f"CKA of scaled inputs: {cka_scaled}")
    assert np.isclose(cka_scaled, 1.0), "CKA should be invariant to scaling"

    X = np.random.rand(100, 64)
    Y = X + np.random.normal(0, 0.1, X.shape)
    cka_noisy = CKA(X, Y)
    print(f"CKA of noisy inputs: {cka_noisy}")
    assert 0.8 < cka_noisy < 1.0, "CKA of noisy inputs should be high but less than 1"

if __name__ == "__main__":
    test_cka()