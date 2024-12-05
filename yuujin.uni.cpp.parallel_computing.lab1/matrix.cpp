#include "matrix.h"

#include <iomanip>
#include <iostream>

// Умножение матриц в одном потоке
Matrix multiply_matrix_st(const CacheOptimizedMatrix& m1, const CacheOptimizedMatrix& m2)
{
	if (m1.rowMajorMatrix.size.n != m2.rowMajorMatrix.size.m)
	{
		std::cout << "Matrices are not compatable to multiplication!" << std::endl;
		return Matrix{};
	}

	MatrixDimension resultDim = { m1.rowMajorMatrix.size.m, m2.rowMajorMatrix.size.n };
	double* resultMatrixContainter = new double[resultDim.m * resultDim.n];

	for (int i = 0; i < resultDim.m; i++)
	{
		for (int j = 0; j < resultDim.n; j++)
		{
			double sum = 0.0;
			for (int k = 0; k < m1.rowMajorMatrix.size.n; k++)
			{
				sum += m1.rowMajorMatrix.container[i * m1.rowMajorMatrix.size.n + k] * m2.colMajorMatrix.container[j * m1.rowMajorMatrix.size.n + k];
			}

			resultMatrixContainter[i * resultDim.n + j] = sum;
		}
	}

	return { resultMatrixContainter, resultDim };
}

Matrix generate_matrix(MatrixDimension dim)
{
	int size = dim.m * dim.n;
	double* container = new double[size];

	for (int i = 0; i < size; i++)
	{
		container[i] = rand() / (double)RAND_MAX;
	}

	return { container, dim };
}

void print_matrix(const Matrix& matrix)
{
	std::cout << "{" << std::endl;
	for (int i = 0; i < matrix.size.m; i++)
	{
		std::cout << " { ";
		for (int j = 0; j < matrix.size.n; j++)
		{
			double item = get_matrix_item(matrix, i, j);
			std::cout << std::fixed << std::setprecision(4) << item;
			if (j != matrix.size.n - 1)
				std::cout << ", ";
		}

		if (i != matrix.size.m - 1)
			std::cout << " },\n";
		else
			std::cout << " }\n";
	}
	std::cout << "}" << std::endl;
}

double get_matrix_item(const Matrix& m, int m_index, int n_index)
{
	if (m_index > m.size.m - 1)
		return 0;

	if (n_index > m.size.n - 1)
		return 0;

	return m.container[m_index * m.size.n + n_index];
}

void set_matrix_item(const Matrix& m, int m_index, int n_index, double item)
{
	if (m_index > m.size.m - 1)
		return;

	if (n_index > m.size.n - 1)
		return;

	m.container[m_index * m.size.n + n_index] = item;
}


CacheOptimizedMatrix make_cache_matrix(const Matrix& matrix)
{
	double* newContainer = new double[matrix.size.n * matrix.size.m];
	Matrix colMajorMatrix{ newContainer, {matrix.size.n, matrix.size.m} };

	for (int i = 0; i < matrix.size.m; i++)
	{
		for (int j = 0; j < matrix.size.n; j++)
		{
			double item = get_matrix_item(matrix, i, j);
			set_matrix_item(colMajorMatrix, j, i, item);
		}
	}

	return { matrix, colMajorMatrix };
}