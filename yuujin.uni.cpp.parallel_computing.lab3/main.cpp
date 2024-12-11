#include <iomanip>
#include <iostream>
#include <chrono>
#include <thread>
#include <omp.h>

#include "common.h"

using namespace std;
using namespace std::chrono;

double calc_integral_st(double start, double end, double step);
double calc_integral_mt(double start, double end, double step);
double calc_integral_omp(double start, double end, double step);

int main()
{
	double start = 0.0;
	double end = 100.0;
	double step = 1e-6;

	auto startSingleThread = high_resolution_clock::now();
	double m3 = calc_integral_st(start, end, step);
	auto endSingleThread = high_resolution_clock::now();

	auto startMultiThreaded = high_resolution_clock::now();
	double m4 = calc_integral_mt(start, end, step);
	auto endMultiThreaded = high_resolution_clock::now();

	auto startOpenMP = high_resolution_clock::now();
	double m5 = calc_integral_omp(start, end, step);
	auto endOpenMP = high_resolution_clock::now();

	cout << "Single-threaded value: " << m3 << endl;
	cout << "Milti-threaded value : " << m4 << endl;
	cout << "OpenMP-threaded value: " << m5 << endl;

	cout << endl;


	auto durationSt = duration_cast<microseconds>(endSingleThread - startSingleThread);
	auto durationMt = duration_cast<microseconds>(endMultiThreaded - startMultiThreaded);
	auto durationMp = duration_cast<microseconds>(endOpenMP - startOpenMP);

	cout << "Single-threaded duration: " << durationSt.count() / 1000.0 << "ms" << endl;
	cout << "Milti-threaded duration : " << durationMt.count() / 1000.0 << "ms (x" << fixed << setprecision(2) << ((double)durationSt.count() / durationMt.count()) << " speed up)" << endl;
	cout << "OpenMP-threaded duration: " << durationMp.count() / 1000.0 << "ms (x" << fixed << setprecision(2) << ((double)durationSt.count() / durationMp.count()) << " speed up)" << endl;

	return 0;
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