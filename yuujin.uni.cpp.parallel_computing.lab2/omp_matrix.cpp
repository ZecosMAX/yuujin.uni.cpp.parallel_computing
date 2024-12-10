#include "omp_matrix.h"

#include <iostream>
#include <omp.h>

Matrix make_upper_triangle_matrix_omp(const Matrix& matrix) 
{
	Matrix result = { new double[matrix.size.m * matrix.size.n], { matrix.size.m, matrix.size.n } };
	memcpy(result.container, matrix.container, sizeof(double) * matrix.size.m * matrix.size.n);

	//for every row
	for (int row = 0; row < result.size.m - 1; row++)
	{
		// take the first item (assuming the previous items) are 0
		double item = result.container[row * result.size.n + row]; //get_matrix_item(result, row, row);

		// ...and subtract the [row] from every next row in a way, that makes the first coeffitient 0
#pragma omp parallel for
		for (int changeRow = row + 1; changeRow < result.size.m; changeRow++)
		{
			double item_to_null = result.container[changeRow * result.size.n + row]; // get_matrix_item(result, changeRow, row);
			double coeffitient = item_to_null / item;

#pragma omp parallel for
#pragma vector always
			for (int col = 0; col < result.size.n; col++)
			{
				result.container[changeRow * result.size.n + col] -= coeffitient * result.container[row * result.size.n + col];
			}
		}
	}
	return { result.container, result.size };
}

Matrix solve_gauss_omp(const Matrix& matrix) 
{
	Matrix triangle_matrix = make_upper_triangle_matrix_omp(matrix);

	Matrix result = { new double[matrix.size.m * matrix.size.n], { matrix.size.m, matrix.size.n } };
	memcpy(result.container, triangle_matrix.container, sizeof(double) * matrix.size.m * matrix.size.n);

	//for every row
	for (int row = result.size.m - 1; row >= 0; row--)
	{
		// take the first item (assuming the previous items) are 0
		double item = result.container[row * result.size.n + row]; // get_matrix_item(result, row, row);

		// ...and subtract the [row] from every next row in a way, that makes the first coeffitient 0
#pragma omp parallel for
		for (int changeRow = row - 1; changeRow >= 0; changeRow--)
		{
			double item_to_null = result.container[changeRow * result.size.n + row]; // get_matrix_item(result, changeRow, row);
			double coeffitient = item_to_null / item;

#pragma omp parallel for
#pragma vector always
			for (int col = 0; col < result.size.n; col++)
			{
				result.container[changeRow * result.size.n + col] -= coeffitient * result.container[row * result.size.n + col];
			}
		}

		// normalize current row
#pragma omp parallel for
#pragma vector always
		for (int col = 0; col < result.size.n; col++)
		{
			result.container[row * result.size.n + col] /= item;
		}

		//std::cout << "Iteration #" << row + 1 << " = " << std::endl;
		//print_matrix(result);
	}
	return { result.container, result.size };
}