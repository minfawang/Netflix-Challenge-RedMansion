#include <omp.h>
#include "types.hpp"

int main(int argc, char * argv[]) {
	timer tmr;
	const int N = 100000000;
	int *a, *b, *c;
	a = new int[N];
	b = new int[N];
	c = new int[N];

	memset((void *)a, 0x0f, N * sizeof(int));
	memset((void *)b, 0x1f, N * sizeof(int));

	tmr.tic();

#pragma omp parallel for num_threads(4)
	for (int i = 0; i < N; i++) {
		c[i] = a[i] + b[i];
		for (int j = 0; j < 10; j++) {
			c[i] += a[i] + b[i];
		}
	}

	tmr.toc();
	system("pause");
}