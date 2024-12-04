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
#include <omp.h>

typedef struct MatrixDimension {
	int m; // vertical size aka row count
	int n; // horizontal size aka column count
};

typedef struct Matrix {
	double* container;
	MatrixDimension size;
};

typedef struct CacheOptimizedMatrix {
	Matrix rowMajorMatrix;
	Matrix colMajorMatrix;
};

Matrix generate_matrix(MatrixDimension);
CacheOptimizedMatrix make_cache_matrix(const Matrix&);
double get_matrix_item(const Matrix&, int, int);
void set_matrix_item(const Matrix&, int, int, double);
void print_matrix(const Matrix&);

Matrix multiply_matrix_st(const CacheOptimizedMatrix&, const CacheOptimizedMatrix&);
Matrix multiply_matrix_mt(const CacheOptimizedMatrix&, const CacheOptimizedMatrix&);

int max_threads_count = 16;

int main()
{
	srand(time(NULL));

	Matrix m1 = generate_matrix({ 1500, 1500 });
	Matrix m2 = generate_matrix({ 1500, 1500 });

	CacheOptimizedMatrix c2 = make_cache_matrix(m2);
	CacheOptimizedMatrix c1 = make_cache_matrix(m1);

	auto startSingleThread = std::chrono::high_resolution_clock::now();
	Matrix m3 = multiply_matrix_st(c1, c2);
	auto endSingleThread = std::chrono::high_resolution_clock::now();

	auto startMultiThread = std::chrono::high_resolution_clock::now();
	Matrix m4 = multiply_matrix_mt(c1, c2);
	auto endMultiThread = std::chrono::high_resolution_clock::now();

	auto durationSt = std::chrono::duration_cast<std::chrono::microseconds>(endSingleThread - startSingleThread);
	auto durationMt = std::chrono::duration_cast<std::chrono::microseconds>(endMultiThread - startMultiThread);
	
	std::cout << "Single-threaded duration: " << durationSt.count() / 1000.0 << "ms" << std::endl;
	std::cout << "Multi-threaded duration : " << durationMt.count() / 1000.0 << "ms (x" << std::fixed << std::setprecision(2) << ((double)durationSt.count() / durationMt.count()) << " speed up)" << std::endl;
}

// функции подготовки данных

Matrix generate_matrix(MatrixDimension dim) 
{
	int size = dim.m * dim.n;
	double* container = new double[size];

	for (int i = 0; i < size; i++)
	{
		container[i] = rand() / (double)RAND_MAX;
	}

	return { container, dim };
}

void print_matrix(const Matrix& matrix)
{
	std::cout << "{" << std::endl;
	for (int i = 0; i < matrix.size.m; i++)
	{
		std::cout << " { ";
		for (int j = 0; j < matrix.size.n; j++)
		{
			double item = get_matrix_item(matrix, i, j);
			std::cout << std::fixed << std::setprecision(4) << item;
			if (j != matrix.size.n - 1)
				std::cout << ", ";
		}

		if (i != matrix.size.m - 1)
			std::cout << " },\n";
		else
			std::cout << " }\n";
	}
	std::cout << "}" << std::endl;
}

double get_matrix_item(const Matrix& m, int m_index, int n_index) 
{
	if (m_index > m.size.m - 1)
		return 0;

	if (n_index > m.size.n - 1)
		return 0;

	return m.container[m_index * m.size.n + n_index];
}

void set_matrix_item(const Matrix& m, int m_index, int n_index, double item)
{
	if (m_index > m.size.m - 1)
		return;

	if (n_index > m.size.n - 1)
		return;

	m.container[m_index * m.size.n + n_index] = item;
}


CacheOptimizedMatrix make_cache_matrix(const Matrix& matrix)
{
	double* newContainer = new double[matrix.size.n * matrix.size.m];
	Matrix colMajorMatrix{ newContainer, {matrix.size.n, matrix.size.m} };

	for (int i = 0; i < matrix.size.m; i++)
	{
		for (int j = 0; j < matrix.size.n; j++)
		{
			double item = get_matrix_item(matrix, i, j);
			set_matrix_item(colMajorMatrix, j, i, item);
		}
	}

	return { matrix, colMajorMatrix };
}

// Сама лабораторная работа начинается тут

// Умножение матриц в одном потоке
Matrix multiply_matrix_st(const CacheOptimizedMatrix& m1, const CacheOptimizedMatrix& m2) 
{
	if (m1.rowMajorMatrix.size.n != m2.rowMajorMatrix.size.m)
	{
		std::cout << "Matrices are not compatable to multiplication!" << std::endl;
		return Matrix{};
	}

	MatrixDimension resultDim = { m1.rowMajorMatrix.size.m, m2.rowMajorMatrix.size.n };
	double* resultMatrixContainter = new double[resultDim.m * resultDim.n];

	for (int i = 0; i < resultDim.m; i++)
	{
		for (int j = 0; j < resultDim.n; j++)
		{
			double sum = 0.0;
			for (int k = 0; k < m1.rowMajorMatrix.size.n; k++)
			{
				sum += m1.rowMajorMatrix.container[i * resultDim.m + k] * m2.colMajorMatrix.container[j * m2.colMajorMatrix.size.n + k];
			}

			resultMatrixContainter[i * resultDim.m + j] = sum;
		}
	}

	return { resultMatrixContainter, resultDim };
}

// Умножение матриц в нескольких потоках (параллельно)
typedef struct sum_task_result {
	double sum;
	int i;
	int j;
};

double calculate_sum_task(double* m1RowArr, double* m2ColArr, int m, int n, int p, int i, int j)
{
	double sum = 0.0;
	for (int k = 0; k < n; k++)
	{
		sum += m1RowArr[i * m + k] * m2ColArr[j * p + k];
	}

	return sum;
}

Matrix multiply_matrix_mt(const CacheOptimizedMatrix& m1, const CacheOptimizedMatrix& m2)
{
	if (m1.rowMajorMatrix.size.n != m2.rowMajorMatrix.size.m)
	{
		std::cout << "Matrices are not compatable to multiplication!" << std::endl;
		return Matrix{};
	}

	MatrixDimension resultDim = { m1.rowMajorMatrix.size.m, m2.rowMajorMatrix.size.n };
	double* resultMatrixContainter = new double[resultDim.m * resultDim.n];

	for (int i = 0; i < resultDim.m; i++)
	{
		#pragma omp parallel for
		for (int j = 0; j < resultDim.n; j++)
		{
			double sum = calculate_sum_task(
				m1.rowMajorMatrix.container, 
				m2.colMajorMatrix.container, 
				m1.rowMajorMatrix.size.m, 
				m1.rowMajorMatrix.size.n, 
				m2.colMajorMatrix.size.n,
				i,
				j);

			resultMatrixContainter[i * resultDim.m + j] = sum;
		}
	}

	return { resultMatrixContainter, resultDim };
}
