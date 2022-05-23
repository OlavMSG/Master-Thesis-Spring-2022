# Master-Thesis-Spring-2022
Code TMA4900 Industrial Mathematics, Master’s Thesis

Based on [Specialization-Project-fall-2021](https://github.com/OlavMSG/Specialization-Project-fall-2021), [LICENCE copy](LICENSE-Specialization-Project-fall-2021).

Project depends on [Matrix LSQ](https://github.com/OlavMSG/matrix-lsq-1)



### Some known limitations
#### In [POD](https://github.com/OlavMSG/Master-Thesis-Spring-2022/blob/main/src/fem_quadrilateral/pod.py#L42) 
eigh - scipy.linalg.eigh is not compliantly stable and it can also be quite slow
fractional_matrix_power - scipy.linalg.fractional_matrix_power is really slow (is in the else). 
At least slower than eigh, sparsity of a_mean is lost in input where a_mean.A is called giving the np.array 
and the unction can use much RAM if a_mean is large ~ 10_000 x 10_000.
Testing if case against each other on case with n_free = 12_960 and ns = 15_625
- gives 3:44 in eigh for corr_mat (times in mm:ss)
- gives 20:02 in fractional_matrix_power and 2:09 in eigh for k_mat
if ns <= n_free - is not necessary because corr_mat and k_mat have the same eigenvalues, but it gives the smallest matrix between corr_mat and k_mat
#### In [_sym_mls_params_setup](https://github.com/OlavMSG/Master-Thesis-Spring-2022/blob/main/src/fem_quadrilateral/fem_quadrilateral_solvers.py#L341) of QuadrilateralSolver
The construction and thereby the evaluation of the Legendre Polynomials is not optimal, however, rewriting this is out of scoop for the current thesis.