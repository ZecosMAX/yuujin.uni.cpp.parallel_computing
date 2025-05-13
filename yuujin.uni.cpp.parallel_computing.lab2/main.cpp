// 2.	«Решение уравнений методом Гаусса»
// Принцип лабораторной работы аналогичен первой: 
// составление и проверка алгоритма решения уравнений методом Гаусса 
// на примерах с малым количеством уравнений(2 - 3), 
// сравнение времени выполнения программы в разных режимах 
// на примере с большим количеством уравнений(от 500).

#include <iomanip>
#include <iostream>
#include <chrono>
#include <thread>
#include <omp.h>
#include <mpi.h>

#include "matrix.h"
#include "thread_matrix.h"
#include "omp_matrix.h"
#include "mpi_matrix.h"

using namespace std;
using namespace std::chrono;

bool is_mpiexec_mode();

int main()
{
	int mpi_rank = 0;
	int mpi_size = 0;

	if (is_mpiexec_mode())
	{
		MPI_Init(NULL, NULL); // Инициализация MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

		if (mpi_rank == 0)
		{
			cout << "\033[1;32mProgram was launched via mpiexec utility -- MPI code will run\033[0m" << endl;
			cout << "\033[1;32m " << mpi_size << " Processes were launched" << "\n\033[0m" << endl;
		}
	}
	else
	{
		cout << "\033[1;33mProgram was launched WITHOUT mpiexec utility! -- MPI code NOT will run\n\033[0m" << endl;
	}

	bool printResults = true;

	srand(time(NULL));
	double numbers[12] = { 2, 1, -1, 8, -3, -1, 2, -11, -2, 1, 2, -3};
	int size = 5000;
	Matrix m1 = generate_matrix({ size, size + 1 }); //{ numbers, { 3, 4 } };

	if (m1.size.m > 5 || m1.size.n > 6)
		printResults = false;

	auto startSingleThread = high_resolution_clock::now();
	auto endSingleThread = high_resolution_clock::now();
	auto startMultiThreaded = high_resolution_clock::now();
	auto endMultiThreaded = high_resolution_clock::now();
	auto startOpenMP = high_resolution_clock::now();
	auto endOpenMP = high_resolution_clock::now();
	auto startMpi = high_resolution_clock::now();
	auto endMpi = high_resolution_clock::now();

	Matrix m3;
	Matrix m4;
	Matrix m5;
	Matrix m6;


	if (mpi_rank == 0)
	{
		cout << "Running single thread..." << endl;
		startSingleThread = high_resolution_clock::now();
		m3 = solve_gauss(m1);
		endSingleThread = high_resolution_clock::now();

		cout << "Running classic multi-thread..." << endl;
		startMultiThreaded = high_resolution_clock::now();
		m4 = solve_gauss_mt(m1);
		endMultiThreaded = high_resolution_clock::now();

		cout << "Running OMP multi-thread..." << endl;
		startOpenMP = high_resolution_clock::now();
		m5 = solve_gauss_omp(m1);
		endOpenMP = high_resolution_clock::now();
	}


	if (is_mpiexec_mode())
	{
		MPI_Barrier(MPI_COMM_WORLD);

		if (mpi_rank == 0)
			cout << "Running MPI multi-process..." << endl;

		startMpi = high_resolution_clock::now();
		m6 = solve_gauss_mpi(m1);
		endMpi = high_resolution_clock::now();
	}

	if (mpi_rank == 0)
	{
		if (printResults)
		{
			print_matrix(m1);
			cout << " -> " << endl;
			print_matrix(m3);
			cout << "=" << endl;
			print_matrix(m4);
			cout << "=" << endl;
			print_matrix(m5);
			cout << "=" << endl;
			print_matrix(m6);
			cout << endl;
			cout << endl;
			cout << endl;
		}


		auto durationSt = duration_cast<microseconds>(endSingleThread - startSingleThread);
		auto durationMt = duration_cast<microseconds>(endMultiThreaded - startMultiThreaded);
		auto durationMp = duration_cast<microseconds>(endOpenMP - startOpenMP);
		auto durationMpi = duration_cast<microseconds>(endMpi - startMpi);

		cout << "Single-threaded duration: " << durationSt.count() / 1000.0 << "ms" << endl;
		cout << "Milti-threaded duration : " << durationMt.count() / 1000.0 << "ms (x" << fixed << setprecision(2) << ((double)durationSt.count() / durationMt.count()) << " speed up)" << endl;
		cout << "OpenMP-threaded duration: " << durationMp.count() / 1000.0 << "ms (x" << fixed << setprecision(2) << ((double)durationSt.count() / durationMp.count()) << " speed up)" << endl;
		cout << "MPI-processed duration  : " << durationMpi.count() / 1000.0 << "ms (x" << fixed << setprecision(2) << ((double)durationSt.count() / durationMpi.count()) << " speed up)" << endl;
	}

	if (is_mpiexec_mode())
		MPI_Finalize();

	return 0;
}

bool is_mpiexec_mode() {
	// Check common MPI environment variables
	const char* env_vars[] = {
		"PMI_SIZE",             // Intel MPI, MS-MPI
		"OMPI_COMM_WORLD_SIZE", // OpenMPI
		"MPI_LOCALNRANKS",      // MPICH
		nullptr
	};

	for (int i = 0; env_vars[i] != nullptr; i++) {
		char* value = nullptr;
		size_t len = 0;
		if (_dupenv_s(&value, &len, env_vars[i]) == 0 && value != nullptr) {
			free(value);  // Free allocated memory
			return true;
		}
	}
	return false;
}
