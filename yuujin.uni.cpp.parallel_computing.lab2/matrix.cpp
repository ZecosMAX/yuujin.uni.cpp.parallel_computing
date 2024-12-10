#include "matrix.h"

#include <iomanip>
#include <iostream>

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

double get_matrix_item(const Matrix& m, int row_index, int col_index)
{
	if (row_index > m.size.m - 1)
		return 0;

	if (col_index > m.size.n - 1)
		return 0;

	return m.container[row_index * m.size.n + col_index];
}

void set_matrix_item(const Matrix& m, int row_index, int col_index, double item)
{
	if (row_index > m.size.m - 1)
		return;

	if (col_index > m.size.n - 1)
		return;

	m.container[row_index * m.size.n + col_index] = item;
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

Matrix make_upper_triangle_matrix(const Matrix& matrix)
{
	Matrix result = { new double[matrix.size.m * matrix.size.n], { matrix.size.m, matrix.size.n } };
	memcpy(result.container, matrix.container, sizeof(double) * matrix.size.m * matrix.size.n);

	//for every row
	for (int row = 0; row < result.size.m - 1; row++)
	{
		// take the first item (assuming the previous items) are 0
		double item = get_matrix_item(result, row, row);

		// ...and subtract the [row] from every next row in a way, that makes the first coeffitient 0
		for (int changeRow = row + 1; changeRow < result.size.m; changeRow++)
		{
			double item_to_null = get_matrix_item(result, changeRow, row);
			double coeffitient = item_to_null / item;

			for (int col = 0; col < result.size.n; col++)
			{
				result.container[changeRow * result.size.n + col] -= coeffitient * result.container[row * result.size.n + col];
			}
		}
	}
	return { result.container, result.size };
}

Matrix solve_gauss(const Matrix& matrix)
{
	Matrix triangle_matrix = make_upper_triangle_matrix(matrix);

	Matrix result = { new double[matrix.size.m * matrix.size.n], { matrix.size.m, matrix.size.n } };
	memcpy(result.container, triangle_matrix.container, sizeof(double) * matrix.size.m * matrix.size.n);

	//for every row
	for (int row = result.size.m - 1; row >= 0; row--)
	{
		// take the first item (assuming the previous items) are 0
		double item = get_matrix_item(result, row, row);

		// ...and subtract the [row] from every next row in a way, that makes the first coeffitient 0
		for (int changeRow = row - 1; changeRow >= 0; changeRow--)
		{
			double item_to_null = get_matrix_item(result, changeRow, row);
			double coeffitient = item_to_null / item;

			for (int col = 0; col < result.size.n; col++)
			{
				result.container[changeRow * result.size.n + col] -= coeffitient * result.container[row * result.size.n + col];
			}
		}

		// normalize current row
		for (int col = 0; col < result.size.n; col++)
		{
			result.container[row * result.size.n + col] /= item;
		}

		//std::cout << "Iteration #" << row + 1 << " = " << std::endl;
		//print_matrix(result);
	}
	return { result.container, result.size };
}