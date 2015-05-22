#include <armadillo>
using namespace arma;
#ifndef __FANG
#define __FANG



double fang_mul(double * colptr_a, double * colptr_b, int n) {
	// return colptr_a' * colptr_b
	double s1 = 0, s2 = 0;
    int i = 0;
    int j = 1;
	for (; i < n - 1; i+=2, j+=2) {
		s1 += colptr_a[i] * colptr_b[i];
        s2 += colptr_a[j] * colptr_b[j];
	}
    if (i < n) {
        s1 += colptr_a[i] * colptr_b[i];
    }
	return s1 + s2;
}

void fang_add_mul(double * colptr_a, double * colptr_b, double k, int n) {
	//colptr_a += colptr_b * k
    int i = 0;
    int j = 1;
	for (; i < n - 1; i+=2, j+=2) {
		colptr_a[i] += colptr_b[i] * k;
        colptr_a[j] += colptr_b[j] * k;
	}
    if (i < n) {
        colptr_a[i] += colptr_b[i] * k;
    }
}

void fang_add_mul2(double * colptr_a, double * colptr_b, double *k1, double k2, int n) {
    //colptr_a += colptr_b * k
    int i = 0;
    int j = 1;
    for (; i < n - 1; i += 2, j += 2) {
        colptr_a[i] += colptr_b[i] * k1[i] * k2;
        colptr_a[j] += colptr_b[j] * k1[j] * k2;
    }
    if (i < n) {
        colptr_a[i] += colptr_b[i] * k1[i] * k2;
    }
}

void fang_add_mul_rtn(vec& result, double * colptr_a, double * colptr_b, double k, int n) {
	for (int i = 0; i < n; i++) {
		result[i] = colptr_a[i] + colptr_b[i] * k;
	}
}

void fang_positive(double * colptr_a, int n) {
	//colptr_a += colptr_b * k
	for (int i = 0; i < n; i++) {
		if (colptr_a[i] < 0){
			colptr_a[i] = 0;
		}
	}
}

#endif