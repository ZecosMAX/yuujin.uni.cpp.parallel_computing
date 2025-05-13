#include <iomanip>
#include <iostream>
#include <chrono>
#include <thread>
#include <omp.h>
#include <mpi.h>

#include "common.h"

using namespace std;
using namespace std::chrono;

double calc_integral_st(double start, double end, double step);
double calc_integral_mt(double start, double end, double step);
double calc_integral_omp(double start, double end, double step);
double calc_integral_mpi(double start, double end, double step);

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

	double start = 0.0;
	double end = 100.0;
	double step = 1e-7;

	auto startSingleThread = high_resolution_clock::now();
	auto endSingleThread = high_resolution_clock::now();
	auto startMultiThreaded = high_resolution_clock::now();
	auto endMultiThreaded = high_resolution_clock::now();
	auto startOpenMP = high_resolution_clock::now();
	auto endOpenMP = high_resolution_clock::now();
	auto startMpi = high_resolution_clock::now();
	auto endMpi = high_resolution_clock::now();

	double m3;
	double m4;
	double m5;
	double m6;

	if (mpi_rank == 0)
	{
		cout << "Running single thread..." << endl;
		startSingleThread = high_resolution_clock::now();
		m3 = calc_integral_st(start, end, step);
		endSingleThread = high_resolution_clock::now();

		cout << "Running classic multi-thread..." << endl;
		startMultiThreaded = high_resolution_clock::now();
		m4 = calc_integral_mt(start, end, step);
		endMultiThreaded = high_resolution_clock::now();

		cout << "Running OMP multi-thread..." << endl;
		startOpenMP = high_resolution_clock::now();
		m5 = calc_integral_omp(start, end, step);
		endOpenMP = high_resolution_clock::now();
	}

	if (is_mpiexec_mode())
	{
		MPI_Barrier(MPI_COMM_WORLD);

		if (mpi_rank == 0)
			cout << "Running MPI multi-process..." << endl;

		startMpi = high_resolution_clock::now();
		m6 = calc_integral_mpi(start, end, step);
		endMpi = high_resolution_clock::now();
	}

	if (mpi_rank == 0)
	{
		cout << "Single-threaded value: " << fixed << setprecision(8) << m3 << endl;
		cout << "Milti-threaded value : " << fixed << setprecision(8) << m4 << endl;
		cout << "OpenMP-threaded value: " << fixed << setprecision(8) << m5 << endl;
		cout << "MPI-processed value  : " << fixed << setprecision(8) << m6 << endl;

		cout << endl;

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

double calc_integral_st(double start, double end, double step)
{
	double area = 0.0;

	for (double pos = start; pos < end; pos += step)
	{
		area += (function(pos) + function(pos + step)) * 0.5 * step;
	}

	return area;
}

void calc_integral_thread(double start, double end, double step, double threadIndex, double threadSize, double* returnValue)
{
	double area = 0.0;

	double threadStep = (end - start) / threadSize;
	double threadStart = start + threadIndex * threadStep;
	double threadEnd = start + (threadIndex + 1) * threadStep;

	for (double pos = threadStart; pos < threadEnd; pos += step)
	{
		area += (function(pos) + function(pos + step)) * 0.5 * step;
	}

	*returnValue = area;
}

double calc_integral_mt(double start, double end, double step)
{
	auto processor_count = std::thread::hardware_concurrency();

	// if hardware_concurrency return 0 (unable to detect)
	if (processor_count == 0)
		processor_count = 1;

	std::thread* threads = new std::thread[processor_count];
	double* returnValues = new double[processor_count];

	for (int i = 0; i < processor_count; i++)
	{
		threads[i] = std::thread(calc_integral_thread,
			start,
			end,
			step,
			i,
			processor_count,
			&returnValues[i]);
	}

	double area = 0.0;

	for (int i = 0; i < processor_count; i++)
	{
		threads[i].join();
		area += returnValues[i];
	}

	delete[] threads;

	return area;
}

double calc_integral_omp(double start, double end, double step)
{
	double area = 0.0;

	int n = (end - start) / step;
	double rem = (end - start) - (n * step);

#pragma omp parallel for reduction(+:area)
	for (int pos = 0; pos < n; pos += 1)
	{
		area += (function(pos * step) + function((pos + 1) * step)) * 0.5 * step;
	}

	//area += (function(n * step) + function(n * step + rem)) * 0.5 * rem;

	return area;
}

double calc_integral_mpi(double start, double end, double step)
{
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double area = 0;

	int n = (end - start) / step;
	double rem = (end - start) - (n * step);

	double processStep = (end - start) / size;
	double processStart = start + rank * processStep;
	double processEnd = start + (rank + 1) * processStep;

	for (double pos = processStart; pos < processEnd; pos += step)
	{
		area += (function(pos) + function(pos + step)) * 0.5 * step;
	}

	double global_area = 0;
	// Сбор и суммирование результатов на процессе 0 (MPI_Reduce)
	MPI_Reduce(&area, &global_area, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
		return global_area;

	return area;
}