#include "..\includes.hpp"
#include <armadillo>
#include <iomanip>
#include <omp.h>

#ifndef __MF_ESTIMATORS
#define __MF_ESTIMATORS

using namespace arma;

class basic_mf : public estimator_base {
public:
	record_array * ptr_test_data;

	mat U;
	mat V;
	vec A;
	vec B;
	double mu;

	double lambda;
	double learning_rate;
	double learning_rate_per_record;

	unsigned int K;

	unsigned int n_iter;

	basic_mf() {
		ptr_test_data = NULL;
		lambda = 0.04;
		learning_rate = 0.002;
		K = 5;
		n_iter = 14;
	}

	virtual float predict(const record & rcd) const{
		unsigned int i = rcd.user - 1, j = rcd.movie - 1;
		mat result = U.col(i).t() * V.col(j) + A(i) + B(j) + mu;
		return result(0, 0);
	}

	void reshuffle(unsigned int *shuffle_idx, unsigned int n) {
		for (int i = n - 1; i > 0; i--) {
			int j = rand() % (i + 1);
			swap(shuffle_idx[i], shuffle_idx[j]);
		}
	}

	void update(const record & rcd) {
		unsigned int i = rcd.user - 1, j = rcd.movie - 1;
		double pFpX;
		double r_pFpX;
		mat UiVj = U.unsafe_col(i).t() * V.unsafe_col(j);

		pFpX = 2.0 * (rcd.score - mu - (UiVj(0, 0) + A(i) + B(j)));
		r_pFpX = learning_rate_per_record * pFpX;

		// U(:,i) = U(:,i) - rate * gUi; gUi = - pFpX * V(:,j);
		U.col(i) += r_pFpX * V.col(j);

		// V(:,j) = V(:,j) - rate * gVj; gVj = - pFpX * U(:,i);
		V.col(j) += r_pFpX * U.col(i);

		// A(:,i) = A(:,i) - rate * gAi; gAi = - pFpX;
		A(i) += r_pFpX;

		// B(:,j) = B(:,j) - rate * gBj; gBj = - pFpX;
		B(j) += r_pFpX;
	}

	virtual void fit(const record_array & train_data) {
		try {
			unsigned int block_size = train_data.size / 160;
			double shrink = 1 - learning_rate * lambda;
			unsigned int n_user = 0, n_movie = 0;
			unsigned int *shuffle_idx;
			timer tmr;

			tmr.display_mode = 1;
			learning_rate_per_record = learning_rate;

			// Generate shuffle_idx
			cout << train_data.size << endl;

			shuffle_idx = new unsigned int[train_data.size / 10];
			for (int i = 0; i < train_data.size / 10; i++) {
				shuffle_idx[i] = i;
			}

			// Calculate n_user and n_movies
			for (int i = 0; i < train_data.size; i++) {
				if (train_data[i].user > n_user) {
					n_user = train_data[i].user;
				}
				if (train_data[i].movie > n_movie) {
					n_movie = train_data[i].movie;
				}
			}

			// Calculate mu
			unsigned int cnt[6];
			long long s;

			for (int i = 0; i < 6; i++) {
				cnt[i] = 0;
			}

			for (int i = 0; i < train_data.size; i++) {
				cnt[int(train_data[i].score)]++;
			}

			s = 0;
			for (int i = 0; i < 6; i++) {
				s += cnt[i] * i;
			}

			mu = 1.0 * s / train_data.size;


			// Reshape the matrix based on n_user and n_movie
			U.set_size(K, n_user);
			V.set_size(K, n_movie);
			A.set_size(n_user);
			B.set_size(n_movie);

			U.fill(fill::randu);
			V.fill(fill::randu);
			A.fill(fill::randu);
			B.fill(fill::randu);

			
			for (int i_iter = 0; i_iter < n_iter; i_iter++) {
				mat oU(U);
				mat oV(V);
				vec oA(A);
				vec oB(B);

				tmr.tic();
				cout << "Iter\t" << i_iter << '\t';

				// Reshuffle first
				reshuffle(shuffle_idx, train_data.size / 10);

#pragma omp parallel for num_threads(8)
				for (int i = 0; i < train_data.size / 10; i++) {
					

					for (int j = 0; j < 10; j++) {
						unsigned int index = shuffle_idx[i] * 10 + j;
						if (index < train_data.size) {
							record rcd = train_data[index];
							update(rcd);
						}
					}

					if (i % block_size == 0) {
						cout << '.';										
					}
				}
				if (ptr_test_data != NULL) {
					vector<float> result = this->predict_list(*ptr_test_data);
					cout << fixed;
					cout << setprecision(5);
					cout << '\t' << MSE(*ptr_test_data, result);
				}


				cout << '\t';
				tmr.toc();

				// Regularization
				//U *= shrink;
				//V *= shrink;
				//A *= shrink;
				//B *= shrink;
				U -= lambda * oU;
				V -= lambda * oV;
				A -= lambda * oA;
				B -= lambda * oB;

				learning_rate *= 0.95;
				lambda *= 0.95;
			}
			delete[]shuffle_idx;
		}
		catch (std::bad_alloc & ba) {
			cout << "bad_alloc caught: " << ba.what() << endl;
			system("pause");
		}
	}

};

#endif
