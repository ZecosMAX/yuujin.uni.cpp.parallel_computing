#include "mpi_matrix.h"

#include <iostream>
#include <mpi.h>


// Умножение матриц в нескольких потоках с использованием стандартных потоков (параллельно)
// Хотя в данном случае "потоками" выступают отдельные процессы MPI, код от "многопоточной версии" не отличается
void calculate_part_mpi(double* m1RowArr, double* m2ColArr, double* rRowArr, int n_start, int n_end, int m, int p)
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

Matrix multiply_matrix_mpi(const CacheOptimizedMatrix& m1, const CacheOptimizedMatrix& m2)
{
	// главная идея программирования под MPI заключается в том, что все процессы запускают один и тот же код
	// необходмио согласовать ветвения так, чтобы главный процесс (rank == 0) выполнял помимо вычислений
	// распределение данные, а так-же ввод/вывод из консольного окна, а остальные процесс только вычисление и получение данных

	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	bool terminate = false;
	

	MPI_Bcast(
		&terminate,
		1,
		MPI_C_BOOL,
		0,
		MPI_COMM_WORLD);

	if (m1.rowMajorMatrix.size.n != m2.rowMajorMatrix.size.m && rank == 0)
	{
		// вывод в консоль всегда только с главного процесса
		std::cout << "Matrices are not compatable to multiplication!" << std::endl;
		return { };
	}

	MatrixDimension resultDim;

	if (rank == 0)
		resultDim = { m1.rowMajorMatrix.size.m, m2.rowMajorMatrix.size.n };

	MPI_Bcast(
		&resultDim,			// хранилище данных
		sizeof(resultDim),	// количество элементов
		MPI_BYTE,			// тип данных
		0,					// Кто главный - в данном случае 0
		MPI_COMM_WORLD);	// набор процессов - все

	int overallSize = resultDim.m * resultDim.n;
	double* resultMatrixContainter = new double[overallSize];
	double* localMatrixContainter = new double[overallSize];

	// снова, выделяем память и организуем данные только на главном процессе
	int* rows_count_data = new int[size * 6];
	int* offsets = new int[size]; // row_start * p + 0;
	int* counts = new int[size]; // (row_end - row_start)* p;
	if (rank == 0)
	{
		double rowsPerProcess = (double)resultDim.m / size; // -- 1000 / 16 = 62.5; avg rows per thread
		int minimalRowsPerProcess = rowsPerProcess; // floor the 62.5 to 62, so every single thread will calculate at least 62 rows
		int remainderRows = resultDim.m - (minimalRowsPerProcess * size); // 1000 - (62 * 16) = 8, how many rows are left

		// заполняем данные для процессов, кто сколько будет выполнять строк
		for (int i = 0, rowStartCounter = 0; i < size; i++)
		{
			int rowCount = minimalRowsPerProcess;

			if (i < remainderRows)
				rowCount += 1;

			rows_count_data[6 * i + 0] = rowStartCounter;				// указываем для какого процесса с какой строки начинать
			rows_count_data[6 * i + 1] = rowStartCounter + rowCount;	// указываем для какого процесса до какой строки считать
			rows_count_data[6 * i + 2] = m1.rowMajorMatrix.size.n;		// параметры размеров
			rows_count_data[6 * i + 3] = m2.rowMajorMatrix.size.n;		// параметры размеров
			rows_count_data[6 * i + 4] = m1.rowMajorMatrix.size.m * m1.rowMajorMatrix.size.n;
			rows_count_data[6 * i + 5] = m2.colMajorMatrix.size.m * m2.colMajorMatrix.size.n;

			offsets[i] = rowStartCounter * m2.rowMajorMatrix.size.n + 0;
			counts[i] = rowCount * m2.rowMajorMatrix.size.n;

			rowStartCounter += rowCount;
		}
	}

	// Далее, синхронизируем задачи по процессам
	MPI_Bcast(
		rows_count_data,	// хранилище данных
		size * 6,			// количество элементов
		MPI_INT,			// тип данных
		0,					// Кто главный - в данном случае 0
		MPI_COMM_WORLD);	// набор процессов - все

	int row_start	= rows_count_data[6 * rank + 0];
	int row_end		= rows_count_data[6 * rank + 1];
	int m			= rows_count_data[6 * rank + 2];
	int p			= rows_count_data[6 * rank + 3];
	int m1rsize		= rows_count_data[6 * rank + 4];
	int m2csize		= rows_count_data[6 * rank + 5];

	double* m1rContainter;
	double* m2cContainter;

	if (rank == 0)
	{
		m1rContainter = m1.rowMajorMatrix.container;
		m2cContainter = m2.colMajorMatrix.container;
	}
	else
	{
		m1rContainter = new double[m1rsize];
		m2cContainter = new double[m2csize];
	}

	// т.к. перемножение матриц так или иначе требует значение "всех элементов" матрицы
	// даже при расчёте одной строки, надо передать данные матрицы первого процесса всем
	MPI_Bcast(
		m1rContainter,		// хранилище данных
		m1rsize,			// количество элементов
		MPI_DOUBLE,			// тип данных
		0,					// Кто главный - в данном случае 0
		MPI_COMM_WORLD);	// набор процессов - все

	MPI_Bcast(
		m2cContainter,		// хранилище данных
		m2csize,			// количество элементов
		MPI_DOUBLE,			// тип данных
		0,					// Кто главный - в данном случае 0
		MPI_COMM_WORLD);	// набор процессов - все

	// теперь можно пойти и выполнить части данных
	calculate_part_mpi(
		m1rContainter,
		m2cContainter,
		localMatrixContainter,
		row_start,
		row_end,
		m,
		p
	);

	// вычисленные данные хранятся в локальном контейнере в следующем формате
	// rank = 0
	// [0][1][2][3][4]...
	// [X][X][X][X][X]...
	//
	// rank = 1
	// [X][X][X][X][X]...
	// [5][6][7][8][9]...
	//
	// соответственно, необходимо синхронизировать эти данные, и отправить в главный процесс
	// так как данные в памяти разложены по строкам друг-за-другом, а разбиения "каждая-другая-строка" нет
	// так как алгоритм пытается оптимизировать по кэшу, то реально, вычисленные данные всегда находятся
	// в одном куске непрерывной памяти, и его можно соответственно передать без циклов
	// начало памяти будет первый элемент соответствующей номера строки (row_start * p (размер строки) + 0)
	// а конец памяти соответственно будет + кол-во элементов
	// а количество элементов это (row_end - row_start) * p; пр. (5 - 2) = 3; 3 * 64 = 192.
    

	// Сбор результатов на процессе 0
	if (true)
	{

		MPI_Gatherv(
			&localMatrixContainter[row_start * p + 0], (row_end - row_start) * p, MPI_DOUBLE, // Отправляемые данные
			resultMatrixContainter, counts, offsets, MPI_DOUBLE,  // Буфер для приема
			0, MPI_COMM_WORLD);

		if(rank == 0)
			std::cout << "Process [" << rank << "/" << size << "]: Results: " << resultMatrixContainter[0] << std::endl;

		//MPI_Barrier(MPI_COMM_WORLD);
	}

	if (rank != 0)
	{
		delete[] m1rContainter;			// очистим память, мы же молодцы?
		delete[] m2cContainter;			// очистим память, мы же молодцы?
	}
	delete[] counts;				// очистим память, мы же молодцы?
	delete[] offsets;				// очистим память, мы же молодцы?
	delete[] localMatrixContainter; // очистим память, мы же молодцы?
	delete[] rows_count_data;		// очистим память, мы же молодцы?

	if (rank == 0)
		return { resultMatrixContainter, resultDim }; // resultMatrixContainter, resultDim
	else
		return { };
}