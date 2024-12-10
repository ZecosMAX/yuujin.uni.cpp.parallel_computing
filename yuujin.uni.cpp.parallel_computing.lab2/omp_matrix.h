#pragma once
#include "matrix.h"

Matrix make_upper_triangle_matrix_omp(const Matrix&);
Matrix solve_gauss_omp(const Matrix&);