#include "thread_matrix.h"

#include <iostream>
#include <thread>

// Умножение матриц в нескольких потоках с использованием стандартных потоков (параллельно)
void calculate_part_thread(double* m1RowArr, double* m2ColArr, double* rRowArr, int n_start, int n_end, int m, int p)
{
	for (int i = n_start; i < n_end; i++)
	{
		for (int j = 0; j < p; j++)
		{
			double sum = 0.0;
			for (int k = 0; k < m; k++)
			{
				sum += m1RowArr[i * m + k] * m2ColArr[j * m + k];
			}

			rRowArr[i * p + j] = sum;
		}
	}
}

Matrix multiply_matrix_mt(const CacheOptimizedMatrix& m1, const CacheOptimizedMatrix& m2)
{
	if (m1.rowMajorMatrix.size.n != m2.rowMajorMatrix.size.m)
	{
		std::cout << "Matrices are not compatable to multiplication!" << std::endl;
		return Matrix{};
	}

	MatrixDimension resultDim = { m1.rowMajorMatrix.size.m, m2.rowMajorMatrix.size.n };

	int overallSize = resultDim.m * resultDim.n;
	double* resultMatrixContainter = new double[overallSize];

	auto processor_count = std::thread::hardware_concurrency();

	// if hardware_concurrency return 0 (unable to detect)
	// then set the processor count to the size of resulting data (1 row per thread)
	// and let the sheduler do the hard work, switching the context
	if (processor_count == 0)
		processor_count = resultDim.m;

	// if hardware_concurrency is greater than data size 
	// also set the processor count to the size of resulting data (1 row per thread)
	if (processor_count > resultDim.m)
		processor_count = resultDim.m;

	double rowsPerThread = (double)resultDim.m / processor_count; // -- 1000 / 16 = 62.5; avg rows per thread
	int minimalRowsPerThread = rowsPerThread; // floor the 62.5 to 62, so every single thread will calculate at least 62 rows
	int remainderRows = resultDim.m - (minimalRowsPerThread * processor_count); // 1000 - (62 * 16) = 8, how many rows are left
	
	// make the first 8 (remainder) thread handle 63 rows then
	std::thread* threads = new std::thread[processor_count];
	for (int i = 0, rowStartCounter = 0; i < processor_count; i++)
	{
		int rowCount = minimalRowsPerThread;

		if (i < remainderRows)
			rowCount += 1;

		threads[i] = std::thread(calculate_part_thread,
			m1.rowMajorMatrix.container, 
			m2.colMajorMatrix.container,
			resultMatrixContainter,
			rowStartCounter,
			rowStartCounter + rowCount,
			m1.rowMajorMatrix.size.n,
			m2.rowMajorMatrix.size.n);

		rowStartCounter += rowCount;
	}

	// wait for threads
	for (int i = 0; i < processor_count; i++)
	{
		threads[i].join();
	}

	return { resultMatrixContainter, resultDim };
}