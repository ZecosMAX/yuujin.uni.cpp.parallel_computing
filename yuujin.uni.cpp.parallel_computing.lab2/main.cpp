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

#include "matrix.h"
#include "thread_matrix.h"
#include "omp_matrix.h"

using namespace std;
using namespace std::chrono;

int main()
{
	bool printResults = true;

	srand(time(NULL));
	double numbers[12] = { 2, 1, -1, 8, -3, -1, 2, -11, -2, 1, 2, -3};
	Matrix m1 = generate_matrix({ 1000, 1001 }); //{ numbers, { 3, 4 } };

	if (m1.size.m > 5 || m1.size.n > 6)
		printResults = false;

	auto startSingleThread = high_resolution_clock::now();
	Matrix m3 = solve_gauss(m1);
	auto endSingleThread = high_resolution_clock::now();

	auto startMultiThreaded = high_resolution_clock::now();
	Matrix m4 = solve_gauss_mt(m1);
	auto endMultiThreaded = high_resolution_clock::now();

	auto startOpenMP = high_resolution_clock::now();
	Matrix m5 = solve_gauss_omp(m1);
	auto endOpenMP = high_resolution_clock::now();


	if (printResults)
	{
		print_matrix(m1);
		cout << " -> " << endl;
		print_matrix(m3);
		cout << "=" << endl;
		print_matrix(m4);
		cout << "=" << endl;
		print_matrix(m5);
		cout << endl;
		cout << endl;
		cout << endl;
	}


	auto durationSt = duration_cast<microseconds>(endSingleThread - startSingleThread);
	auto durationMt = duration_cast<microseconds>(endMultiThreaded - startMultiThreaded);
	auto durationMp = duration_cast<microseconds>(endOpenMP - startOpenMP);

	cout << "Single-threaded duration: " << durationSt.count() / 1000.0 << "ms" << endl;
	cout << "Milti-threaded duration : " << durationMt.count() / 1000.0 << "ms (x" << fixed << setprecision(2) << ((double)durationSt.count() / durationMt.count()) << " speed up)" << endl;
	cout << "OpenMP-threaded duration: " << durationMp.count() / 1000.0 << "ms (x" << fixed << setprecision(2) << ((double)durationSt.count() / durationMp.count()) << " speed up)" << endl;

	return 0;
}