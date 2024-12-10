#pragma once

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

Matrix make_upper_triangle_matrix(const Matrix&);
Matrix solve_gauss(const Matrix&);
