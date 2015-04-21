#include "..\includes.hpp"
#include <armadillo>


#ifndef __MF_ESTIMATORS
#define __MF_ESTIMATORS

using namespace arma;

class basic_mf : public estimator_base {
public:
	mat U;
	mat V;
	mat A;
	mat B;
	double mu;

	double lambda;
	double learning_rate;
	double learning_rate_per_record;

	unsigned int K;

	unsigned int n_iter;

	basic_mf() {
		lambda = 0.005;
		learning_rate = 0.002;
		K = 20;
		n_iter = 40;
	}

	virtual float predict(const record & rcd) const{
		unsigned int i = rcd.user - 1, j = rcd.movie - 1;
		mat result = U.col(i).t() * V.col(j) + A.col(i) + B.col(j) + mu;
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
		mat pFpX;
		double r_pFpX;

		pFpX = 2.0 * (rcd.score - mu - (U.col(i).t() * V.col(j) + A.col(i) + B.col(j)));
		r_pFpX = learning_rate_per_record * pFpX(0, 0);

		// U(:,i) = U(:,i) - rate * gUi; gUi = - pFpX * V(:,j);
		U.col(i) += r_pFpX * V.col(j);

		// V(:,j) = V(:,j) - rate * gVj; gVj = - pFpX * U(:,i);
		V.col(j) += r_pFpX * U.col(i);

		// A(:,i) = A(:,i) - rate * gAi; gAi = - pFpX;
		A.col(i) += r_pFpX;

		// B(:,j) = B(:,j) - rate * gBj; gBj = - pFpX;
		B.col(j) += r_pFpX;
	}

	virtual void fit(const record_array & train_data) {
		try {
			double shrink = 1 - learning_rate * lambda;
			unsigned int n_user = 0, n_movie = 0;
			unsigned int *shuffle_idx;
			timer tmr;

			learning_rate_per_record = learning_rate;

			// Generate shuffle_idx
			cout << train_data.size << endl;

			shuffle_idx = new unsigned int[train_data.size];
			for (int i = 0; i < train_data.size; i++) {
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
			U.reshape(K, n_user);
			V.reshape(K, n_movie);
			A.reshape(1, n_user);
			B.reshape(1, n_movie);

			U.fill(fill::randu);
			V.fill(fill::randu);
			A.fill(fill::randu);
			B.fill(fill::randu);

			tmr.tic();
			for (int i_iter = 0; i_iter < n_iter; i_iter++) {
				mat oU(U);
				mat oV(V);
				mat oA(A);
				mat oB(B);
				// Reshuffle first
				reshuffle(shuffle_idx, train_data.size);
				for (int i = 0; i < train_data.size; i++) {
					unsigned int index = shuffle_idx[i];
					record rcd = train_data[index];
					update(rcd);
					if (i % 100000 == 0) {
						cout << i << "   ";
						tmr.toc();						
						tmr.tic();
					}
				}

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
