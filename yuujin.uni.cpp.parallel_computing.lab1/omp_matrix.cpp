#include "omp_matrix.h"

#include <iostream>
#include <omp.h>

// Умножение матриц в нескольких потоках с использованием OpenMP (параллельно)
double calculate_sum_task(double* m1RowArr, double* m2ColArr, int n, int i, int j)
{
	double sum = 0.0;
	for (int k = 0; k < n; k++)
	{
		sum += m1RowArr[i * n + k] * m2ColArr[j * n + k];
	}

	return sum;
}

Matrix multiply_matrix_omp(const CacheOptimizedMatrix& m1, const CacheOptimizedMatrix& m2)
{
	if (m1.rowMajorMatrix.size.n != m2.rowMajorMatrix.size.m)
	{
		std::cout << "Matrices are not compatable to multiplication!" << std::endl;
		return Matrix{};
	}

	MatrixDimension resultDim = { m1.rowMajorMatrix.size.m, m2.rowMajorMatrix.size.n };
	double* resultMatrixContainter = new double[resultDim.m * resultDim.n];

	/*int n = m1.rowMajorMatrix.size.m;
	int m = m1.rowMajorMatrix.size.n;
	int p = m2.rowMajorMatrix.size.n;*/

	for (int i = 0; i < resultDim.m; i++)
	{
#pragma omp parallel for
		for (int j = 0; j < resultDim.n; j++)
		{
			double sum = calculate_sum_task(
				m1.rowMajorMatrix.container,
				m2.colMajorMatrix.container,
				m1.rowMajorMatrix.size.n,
				i,
				j);

			resultMatrixContainter[i * resultDim.n + j] = sum;
		}
	}

	return { resultMatrixContainter, resultDim };
}