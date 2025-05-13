#include "mpi_matrix.h"

#include <iostream>
#include <mpi.h>

void make_upper_triangle_matrix_mpi(double* matrixContainer, MatrixDimension matrixSize, int rank, int size, MPI_Win memoryWindow)
{
	// for every row
	for (int row = 0; row < matrixSize.m - 1; row++)
	{
		// Все процессы должны циклично вычесть подготовленную строку из оставшихся
		double item = matrixContainer[row * matrixSize.n + row];

		for (int changeRow = row + 1; changeRow < matrixSize.m; changeRow++)
		{
			// Пропускаем строку, если она имеет не наш номер
			if (changeRow % size != rank)
				continue;

			double item_to_null = matrixContainer[changeRow * matrixSize.n + row];
			double coeffitient = item_to_null / item;

			for (int col = row; col < matrixSize.n; col++)
			{
				matrixContainer[changeRow * matrixSize.n + col] -= coeffitient * matrixContainer[row * matrixSize.n + col];
			}
		}
		// синхронизируем память
		MPI_Win_fence(0, memoryWindow);

		// чтобы не дай бог какой-либо из процессов пошёл обрабатывать следующую строку и использовал не готовый item
		//MPI_Barrier(MPI_COMM_WORLD);
	}
}

void solve_mpi(double* matrixContainer, MatrixDimension matrixSize, int rank, int size, MPI_Win memoryWindow)
{
	for (int row = matrixSize.m - 1; row >= 0; row--)
	{
		// Все процессы должны циклично вычесть подготовленную строку из оставшихся
		double item = matrixContainer[row * matrixSize.n + row];

		for (int changeRow = row - 1; changeRow >= 0; changeRow--)
		{
			// Пропускаем строку, если она имеет не наш номер
			if (changeRow % size != rank)
				continue;

			double item_to_null = matrixContainer[changeRow * matrixSize.n + row];
			double coeffitient = item_to_null / item;

			for (int col = row; col < matrixSize.n; col++)
			{
				matrixContainer[changeRow * matrixSize.n + col] -= coeffitient * matrixContainer[row * matrixSize.n + col];
			}
		}

		// нормализуем текущую строку, если мы главный процесс
		if (rank == 0)
		{
			for (int col = 0; col < matrixSize.n; col++)
			{
				matrixContainer[row * matrixSize.n + col] /= item;
			}
		}

		// синхронизируем память
		MPI_Win_fence(0, memoryWindow);

		// чтобы не дай бог какой-либо из процессов пошёл обрабатывать следующую строку и использовал не готовый item
		//MPI_Barrier(MPI_COMM_WORLD);
	}
}

Matrix solve_gauss_mpi(const Matrix& matrix)
{
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MatrixDimension matrixSize;

	if (rank == 0)
		matrixSize = matrix.size;

	MPI_Bcast(&matrixSize, sizeof(MatrixDimension), MPI_BYTE, 0, MPI_COMM_WORLD);

	MPI_Win win;
	double* shared_data;

	int a = MPI_Win_allocate_shared(
		(rank == 0) ? sizeof(double) * matrixSize.m * matrixSize.n : 0,
		sizeof(double),
		MPI_INFO_NULL,
		MPI_COMM_WORLD,
		&shared_data, &win
	);

	// Не забыть тыкнуться в общую память
	double* localContainer = nullptr;
	MPI_Aint local_size;
	int disp_unit;
	MPI_Win_shared_query(win, 0, &local_size, &disp_unit, &localContainer);

	memcpy(localContainer, matrix.container, sizeof(double) * matrixSize.m * matrixSize.n);
	MPI_Win_fence(0, win);
	MPI_Barrier(MPI_COMM_WORLD);

	// Делаем матрицу треугольной
	make_upper_triangle_matrix_mpi(localContainer, matrixSize, rank, size, win);

	// Решаем гаусса
	solve_mpi(localContainer, matrixSize, rank, size, win);

	//std::cout << "Process [" << rank << "/" << size << "]: Results: " << std::endl;

	double* returnContainer = nullptr;
	if (rank == 0)
	{
		returnContainer = new double[matrix.size.n * matrix.size.m];
		memcpy(returnContainer, localContainer, sizeof(double) * matrixSize.m * matrixSize.n);
	}

	
	// Cleanup
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Win_fence(0, win);
	MPI_Win_free(&win);

	if (rank == 0)
	{
		return { returnContainer , matrixSize };
	}
	else
		return {};
}