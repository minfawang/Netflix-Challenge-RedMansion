#ifndef __FANG
#define __FANG



double fang_mul(double * colptr_a, double * colptr_b, unsigned int n) {
	// return colptr_a' * colptr_b
	double s = 0;
	for (unsigned int i = 0; i < n; i++) {
		s += colptr_a[i] * colptr_b[i];
	}
	return s;
}

void fang_add_mul(double * colptr_a, double * colptr_b, double k, unsigned int n) {
	//colptr_a += colptr_b * k
	for (unsigned int i = 0; i < n; i++) {
		colptr_a[i] += colptr_b[i] * k;
	}
}

vec fang_add_mul_rtn(double * colptr_a, double * colptr_b, double k, unsigned int n) {
	vec result(n);
	for (unsigned int i = 0; i < n; i++) {
		result[i] = colptr_a[i] + colptr_b[i] * k;
	}
	return result;
}

void fang_positive(double * colptr_a, unsigned int n) {
	//colptr_a += colptr_b * k
	for (unsigned int i = 0; i < n; i++) {
		if (colptr_a[i] < 0){
			colptr_a[i] = 0;
		}
	}
}

#endif