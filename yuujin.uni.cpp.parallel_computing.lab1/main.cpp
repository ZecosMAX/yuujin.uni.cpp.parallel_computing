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

#include "matrix.h"
#include "thread_matrix.h"
#include "omp_matrix.h"

int main()
{
	srand(time(NULL));

	bool printResults = true;

	Matrix m1 = generate_matrix({ 1500, 1300 });
	Matrix m2 = generate_matrix({ 1300, 1500 });
	 
	//Matrix m1 = generate_matrix({ 2, 3 });
	//Matrix m2 = generate_matrix({ 3, 5 });

	if (m1.size.m > 5 || m1.size.n > 5)
		printResults = false;

	CacheOptimizedMatrix c2 = make_cache_matrix(m2);
	CacheOptimizedMatrix c1 = make_cache_matrix(m1);

	auto startSingleThread = std::chrono::high_resolution_clock::now();
	Matrix m3 = multiply_matrix_st(c1, c2);
	auto endSingleThread = std::chrono::high_resolution_clock::now();

	auto startMultiThreaded = std::chrono::high_resolution_clock::now();
	Matrix m4 = multiply_matrix_mt(c1, c2);
	auto endMultiThreaded = std::chrono::high_resolution_clock::now();

	auto startOpenMP = std::chrono::high_resolution_clock::now();
	Matrix m5 = multiply_matrix_omp(c1, c2);
	auto endOpenMP = std::chrono::high_resolution_clock::now();


	if (printResults)
	{
		print_matrix(m1);
		std::cout << "*" << std::endl;
		print_matrix(m2);
		std::cout << "=" << std::endl;
		print_matrix(m3);
		std::cout << "=" << std::endl;
		print_matrix(m4);
		std::cout << "=" << std::endl;
		print_matrix(m5);
	}


	auto durationSt = std::chrono::duration_cast<std::chrono::microseconds>(endSingleThread - startSingleThread);
	auto durationMt = std::chrono::duration_cast<std::chrono::microseconds>(endMultiThreaded - startMultiThreaded);
	auto durationMp = std::chrono::duration_cast<std::chrono::microseconds>(endOpenMP - startOpenMP);
	
	std::cout << "Single-threaded duration: " << durationSt.count() / 1000.0 << "ms" << std::endl;
	std::cout << "Milti-threaded duration : " << durationMt.count() / 1000.0 << "ms (x" << std::fixed << std::setprecision(2) << ((double)durationSt.count() / durationMt.count()) << " speed up)" << std::endl;
	std::cout << "OpenMP-threaded duration: " << durationMp.count() / 1000.0 << "ms (x" << std::fixed << std::setprecision(2) << ((double)durationSt.count() / durationMp.count()) << " speed up)" << std::endl;
}
