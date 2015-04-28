#include <armadillo>
#include <iostream>
#include "types.hpp"
// #include <shark/Unsupervised/RBM/BinaryRBM.h>


#ifndef __RBM_ESTIMATORS
#define __RBM_ESTIMATORS

using namespace arma;


// class user_rating {
// public:
// 	unsigned 
// };


class basic_rbm : public estimator_base {
public:

	cube W; // M * F * K
	mat BV; // K * M
	// mat BH; // K * F
	vec BH; // F

	unsigned int M;
	unsigned int K;
	unsigned int F;
	unsigned int CD_K;

	float lrate; // learning rate


	basic_rbm() {
		K = 5;
		F = 10;
		M = 2000; // TODO: change M to be total number of movies

		W = randu<cube>(K, F, M) / 8.0;
		BV = randu<mat>(K, M) / 8.0;
		// BH = randu<mat>(K, F) / 8.0;
		BH = randu<vec>(F) / 8.0;


		CD_K = 1;
		lrate = 0.0001;


	}

	virtual bool save(const char * file_name) {
		return true;
	}

	virtual bool load(const char * file_name) {
		return true;
	}


	virtual void fit(const record_array & train_data, unsigned int n_iter = 1, bool countinue_fit = false) {
		unsigned int user_id = train_data.data[0].user;
		unsigned int start = 0;
		unsigned int end = 0;
		for (int i = 0; i < train_data.size; i++) {
			record r = train_data.data[i];
			if ((user_id != r.user) || i == train_data.size-1) {
				end = i;
				train((train_data.data+start), user_id, end - start, n_iter);

				user_id = r.user;
				start = i;
			}
		}
	}
	virtual float predict(const record & rcd) const{
		return 0.0;
	}


	void train(const record *data, unsigned int user_id, unsigned int size, unsigned int n_iter = 1) {
		// initialization
		mat V0 = zeros<mat>(K, size);
		mat Vt = zeros<mat>(K, size);
		vec H0 = zeros<vec>(F);
		vec Ht = zeros<vec>(F);


		// set up V0 and Vt based on the input data.
		for (int k = 0; k < K; k++) {
			for (int i = 0; i < size; i++) {
				record r = data[i];
				V0(int(r.score)-1, i) = 1; // score - 1 is the index
				Vt(int(r.score)-1, i) = 1;
			}
		}

		// set up H0 by V -> H
		for (int i = 0; i < size; i++) {
			record r = data[i];

			for (int j = 0; j < F; j++) {
				float bh = BH(j);

				for (int k = 0; k < K; k++) {
					float w = W(k, j, r.movie);
					float v = Vt(k, i);

					H0(j) += w * v + bh;
				}
			}
		}


		// Do the contrastive divergence
		for (int n = 0; n < CD_K; n++) {

			// set H -> 0
			Ht = zeros<vec>(F);

			// positive phase: V -> H
			// TODO: Do we need to normalize H(j)?
			for (int i = 0; i < size; i++) {
				record r = data[i];

				for (int j = 0; j < F; j++) {
					float bh = BH(j);
					for (int k = 0; k < K; k++) {
						float w = W(k, j, r.movie);
						float v = Vt(k, i);

						Ht(j) += w * v + bh;
					}
				}
			}

			// set Vt -> 0
			Vt = zeros<mat>(K, size);


			// negative phase: H -> V
			for (int i = 0; i < size; i++) {
				record r = data[i];
				for (int j = 0; j < F; j++) {
					float h = Ht(j);

					for (int k = 0; k < K; k++) {
						float w = W(k, j, r.movie);

						float bv = BV(k, r.movie);
						Vt(k, i) = h * w + bv;
					}
				}
			}

			// Normalize Vt -> sum_k (Vt(k, i)) = 1
			for (int i = 0; i < size; i++) {
				float sum_k = 0.0;
				for (int k = 0; k < K; k++) {
					sum_k += Vt(k, i);
				}

				for (int k = 0; k < K; k++) {
					Vt(k, i) /= sum_k;
				}
			}

		}

		// update W
		for (int i = 0; i < size; i++) {
			record r = data[i];
			for (int j = 0; j < F; j++) {
				for (int k = 0; k < K; k++) {

					W(k, j, r.movie) += lrate * (H0(j) * V0(k, i) - Ht(j) * Vt(k, i));
				}
			}
		}

		// update BH
		for (int j = 0; j < F; j++) {
			BH(j) += lrate * (H0(j) - Ht(j));
		}

		// update BV
		for (int i = 0; i < size; i++) {
			record r = data[i];
			for (int k = 0; k < K; k++) {
				BV(k, r.movie) += lrate * (V0(k, i) - Vt(k, i));
			}
		}

	}


};


int main () {
	record_array main_data;
	main_data.load("../../data/mini_main.data");

	basic_rbm rbm;

	rbm.fit(main_data);


}





#endif