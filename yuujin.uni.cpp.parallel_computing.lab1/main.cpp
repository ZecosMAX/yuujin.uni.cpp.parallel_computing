//	1.	«Перемножение матриц»
//	В работе необходимо составить и проверить алгоритм перемножения матриц.
//	Для проверки алгоритма целесообразно использовать небольшие матрицы (до 3х3 включительно), 
//	результат перемножения вывести на экран. После успешной проверки задаются большие (1000х1000) 
//	матрицы как входные данные, наполнение матриц случайно и необязательно к выводу на экран.
//	На скриншотах выполнения программы должна быть видна разница 
//	во времени исполнения в последовательном и в параллельном режимах.

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
	srand(time(NULL));

	bool printResults = true;
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

	//Matrix m1 = generate_matrix({ 1500, 1300 });
	//Matrix m2 = generate_matrix({ 1300, 2500 });
	 
	Matrix m1 = generate_matrix({ 3, 2 });
	Matrix m2 = generate_matrix({ 5, 3 });

	if(mpi_rank == 0)
		cout << "Running matrix-sizes of (" << m1.size.m << ", " << m1.size.n << ") and (" << m2.size.m << ", " << m2.size.n << ")" << endl;

	if (m1.size.m > 5 || m1.size.n > 5)
		printResults = false;

	CacheOptimizedMatrix c2;
	CacheOptimizedMatrix c1;

	// Инициализировать данные только в случае если процесс - главный.
	// Свою "часть" данных процессы получат уже при выполнении далее
	if (!is_mpiexec_mode() || mpi_rank == 0)
	{
		c1 = make_cache_matrix(m2);
		c2 = make_cache_matrix(m1);
	}
	
	auto startSingleThread = high_resolution_clock::now();
	auto endSingleThread = high_resolution_clock::now();
	auto startMultiThreaded = high_resolution_clock::now();
	auto endMultiThreaded = high_resolution_clock::now();
	auto startOpenMP = high_resolution_clock::now();
	auto endOpenMP = high_resolution_clock::now();
	auto startMPI = high_resolution_clock::now();
	auto endMPI = high_resolution_clock::now();
	Matrix m3;
	Matrix m4;
	Matrix m5;
	Matrix m6;


	if (!is_mpiexec_mode() || mpi_rank == 0)
	{
		cout << "Running single thread..." << endl;

		startSingleThread = high_resolution_clock::now();
		m3 = multiply_matrix_st(c1, c2);
		endSingleThread = high_resolution_clock::now();

		cout << "Running classic multi-thread..." << endl;

		startMultiThreaded = high_resolution_clock::now();
		m4 = multiply_matrix_mt(c1, c2);
		endMultiThreaded = high_resolution_clock::now();

		cout << "Running OMP multi-thread..." << endl;

		startOpenMP = high_resolution_clock::now();
		m5 = multiply_matrix_omp(c1, c2);
		endOpenMP = high_resolution_clock::now();
	}

	if (is_mpiexec_mode())
	{
		if (mpi_rank == 0)
			cout << "Running MPI multi-process..." << endl;

		// All processes wait here until everyone arrives
		MPI_Barrier(MPI_COMM_WORLD);

		startMPI = high_resolution_clock::now();
		m6 = multiply_matrix_mpi(c1, c2);
		endMPI = high_resolution_clock::now();
	}

	if (!is_mpiexec_mode() || mpi_rank == 0)
	{
		std::cout << "Process [" << mpi_rank << "/" << mpi_size << "]: Results: " << std::endl;

		if (printResults)
		{
			print_matrix(m1);
			cout << "*" << endl;
			print_matrix(m2);
			cout << "=" << endl;
			print_matrix(m3);
			cout << "=" << endl;
			print_matrix(m4);
			cout << "=" << endl;
			print_matrix(m5);
			cout << "=" << endl;
			print_matrix(m6);
		}

		auto durationSt = duration_cast<microseconds>(endSingleThread - startSingleThread);
		auto durationMt = duration_cast<microseconds>(endMultiThreaded - startMultiThreaded);
		auto durationMp = duration_cast<microseconds>(endOpenMP - startOpenMP);

		cout << "Single-threaded duration: " << durationSt.count() / 1000.0 << "ms" << endl;
		cout << "Milti-threaded duration : " << durationMt.count() / 1000.0 << "ms (x" << fixed << setprecision(2) << ((double)durationSt.count() / durationMt.count()) << " speed up)" << endl;
		cout << "OpenMP-threaded duration: " << durationMp.count() / 1000.0 << "ms (x" << fixed << setprecision(2) << ((double)durationSt.count() / durationMp.count()) << " speed up)" << endl;
		cout << "MPI-processed duration  : " << durationMp.count() / 1000.0 << "ms (x" << fixed << setprecision(2) << ((double)durationSt.count() / durationMp.count()) << " speed up)" << endl;
	}

	if(is_mpiexec_mode())
		MPI_Finalize();
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
