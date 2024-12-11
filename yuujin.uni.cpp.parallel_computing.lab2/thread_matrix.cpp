#include "thread_matrix.h"

#include <barrier>
#include <iostream>
#include <string>
#include <syncstream>
#include <thread>
#include <vector>

void calculate_part_triangle_thread(double* resultContainer, double* sourceContainer, int rowSize, int colSize, int threadIndex, int threadSize, int pass)
{
	// because we want to process all rows, we need to iterate over 'every other' item in column
	// Imagine there is 4 cores (4 threads)
	// thread 0 will process rows: [0], [4], [8], ...
	// thread 1 will process rows: [1], [5], [9], ...
	// thread 2 will process rows: [2], [6], [10], ...
	// thread 3 will process rows: [3], [7], [11], ...
	// 
	// all thread will 'cycle' through rows, creating a more uniform load distribution, 
	// than simply dividing every number of rows for a set number of threads
	// (bc that way first batch will finish earlier, and start wait for 'heavier' threads to finish)
	for (int rowIndex = pass + 1 + threadIndex; rowIndex < colSize; rowIndex += threadSize)
	{
		double coeffitient = resultContainer[rowIndex * rowSize + pass] / resultContainer[pass * rowSize + pass];

		for (int col = pass; col < rowSize; col++)
		{
			resultContainer[rowIndex * rowSize + col] -= coeffitient * resultContainer[pass * rowSize + col];
		}
	}
}

void calculate_part_gauss_thread(double* resultContainer, double* sourceContainer, int rowSize, int colSize, int threadIndex, int threadSize, int pass)
{
	for (int rowIndex = pass - 1 - threadIndex; rowIndex >= 0; rowIndex -= threadSize)
	{
		double coeffitient = resultContainer[rowIndex * rowSize + pass] / resultContainer[pass * rowSize + pass];

		for (int col = rowIndex; col < rowSize; col++)
		{
			resultContainer[rowIndex * rowSize + col] -= coeffitient * resultContainer[pass * rowSize + col];
		}
	}
}

Matrix make_upper_triangle_matrix_mt(const Matrix& matrix) 
{
	Matrix result = { new double[matrix.size.m * matrix.size.n], { matrix.size.m, matrix.size.n } };
	memcpy(result.container, matrix.container, sizeof(double) * matrix.size.m * matrix.size.n);

	auto processor_count = std::thread::hardware_concurrency();

	// if hardware_concurrency return 0 (unable to detect)
	if (processor_count == 0)
		processor_count = 1;

	// if hardware_concurrency is greater than data size 
	// set the processor count to the size of resulting data (1 row per thread)
	if (processor_count > matrix.size.m)
		processor_count = matrix.size.m;


	for (size_t row = 0; row < matrix.size.m - 1; row++)
	{
		std::thread* threads = new std::thread[processor_count];

		for (int i = 0; i < processor_count; i++)
		{
			threads[i] = std::thread(calculate_part_triangle_thread,
				result.container,
				matrix.container,
				matrix.size.n,
				matrix.size.m,
				i,
				processor_count,
				row);
		}

		for (int i = 0; i < processor_count; i++)
		{
			threads[i].join();
		}

		delete[] threads;
	}

	return { result.container, result.size };
}

Matrix solve_gauss_mt(const Matrix& matrix) 
{
	Matrix triangleMatrix = make_upper_triangle_matrix_mt(matrix);
	//print_matrix(triangleMatrix);

	auto processor_count = std::thread::hardware_concurrency();
	// if hardware_concurrency return 0 (unable to detect)
	if (processor_count == 0)
		processor_count = 1;

	// if hardware_concurrency is greater than data size 
	// set the processor count to the size of resulting data (1 row per thread)
	if (processor_count > matrix.size.m)
		processor_count = matrix.size.m;

	for (int row = triangleMatrix.size.m - 1; row >= 0; row--)
	{
		std::thread* threads = new std::thread[processor_count];
		for (int i = 0; i < processor_count; i++)
		{
			threads[i] = std::thread(calculate_part_gauss_thread,
				triangleMatrix.container,
				triangleMatrix.container,
				triangleMatrix.size.n,
				triangleMatrix.size.m,
				i,
				processor_count, row);
		}

		for (int i = 0; i < processor_count; i++)
		{
			threads[i].join();
		}

		delete[] threads;
	}

	return { triangleMatrix.container, triangleMatrix.size };
}