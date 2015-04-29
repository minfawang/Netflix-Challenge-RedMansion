#include <armadillo>
#include <iostream>
#include "types.hpp"
// #include <shark/Unsupervised/RBM/BinaryRBM.h>


#ifndef __RBM_ESTIMATORS
#define __RBM_ESTIMATORS

using namespace arma;


class basic_rbm : public estimator_base {
public:

	cube W; // M * F * K
	mat BV; // K * M
	// mat BH; // K * F
	vec BH; // F

	unsigned int N;
	unsigned int M;
	unsigned int K;
	unsigned int F;
	unsigned int CD_K;
	float lrate; // learning rate


	record_array *ptr_test_data;
	record_array *ptr_train_data;




	basic_rbm() {
		K = 5;
		F = 10;
		M = 1778; // TODO: change M to be total number of movies
		N = 458293;

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


		// training stage
		for (int i = 0; i < train_data.size; i++) {
			record r = train_data.data[i];
			if ((user_id != r.user) || i == train_data.size-1) {
				end = i;
				train((train_data.data+start), user_id, end - start, n_iter);

				user_id = r.user;
				start = i;
			}
			if (i % 10000000 == 0) {
				cout << "working on iteration " << i << " ..." << endl;
			}
		}

		cout << "finish training!" << endl;

		// predicting stage
		unsigned int j = 0;
		unsigned int train_start = 0;
		unsigned int train_end = 0;
		unsigned int test_start = 0;
		unsigned int test_end = 0;
		unsigned int train_user = ptr_train_data->data[0].user;
		unsigned int test_user = ptr_test_data->data[0].user;


		for (int i = 0; i < ptr_test_data->size; i++) {

			vec Hu = zeros<vec>(F);
			record r_test = ptr_test_data->data[i];

			if ((test_user != r_test.user) || i == ptr_test_data->size -1) {
				// make prediction of test_user for movies in the test set
				test_end = i;

				// find train_start and train_end
				// record r_train = ptr_train_data->data[j];
				record r_train;
				r_train.user = 0;

				while ((r_train.user <= test_user) && j < ptr_train_data->size) {
					r_train = ptr_train_data->data[j];

					if (r_train.user < test_user) {
						train_start = j + 1;
					}
					j++;
				}

				train_end = j;

				if (ptr_train_data->data[j-1].user == r_test.user) {
					// positive phase to compute Hu
					for (int f = 0; f < F; f++) {
						Hu(f) = BH(f);
					}

					for (int u = train_start; u < train_end; u++) {
						record r_train = ptr_train_data->data[u];
						for (int f = 0; f < F; f++) {
							unsigned int k = int(r_train.score) - 1;
							float w = W(k, f, r_train.movie);
							Hu(f) += w;
						}
					}

					// negative phase to predict score
					for (int u = test_start; u < test_end; u++) {
						record r_test = ptr_test_data->data[u];
						vec rating_probs = zeros<vec>(K);
						float predict_score = 0;

						for (int k = 0; k < K; k++) {
							rating_probs(k) = BV(k, r_test.movie);
						}

						for (int f = 0; f < F; f++) {
							for (int k = 0; k < K; k++) {
								float w = W(k, f, r_test.movie);
								rating_probs(k) += w;
							}
						}

						// normalize rating_probs
						// QUESTION: Is it possible for prob to be less than 0?
						float sum_k = 0;
						for (int k = 0; k < K; k++) {
							sum_k += rating_probs(k);
						}
						for (int k = 0; k < K; k++) {
							rating_probs(k) /= sum_k;
						}

						// update predict score by taking average
						for (int k = 0; k < K; k++) {
							predict_score += (k+1) * rating_probs(k);
						}

						cout << predict_score << "  ";
					}

				} else {
					// TODO: predict all movies to be 3.5
					float predict_score;
					for (int u = test_start; u < test_end; u++) {
						predict_score = 3.5;
					}
				}

				train_start = j;


				test_start = i;
				test_user = r_test.user;
			}
		}

		cout << "finish predicting!" << endl;


		// unsigned int j = 0;
		// user_id = 0;
		// start = 0;
		// end = 0;
		// for (int i = 0; i < ptr_test_data->size; i++) {
		// 	record test_r = ptr_test_data->data[i];
		// 	while (user_id < test_r.user && j < ptr_train_data->size) {
		// 		user_id = ptr_train_data->data[i].user_id;
		// 		j++;
		// 	}
		// 	if (user_id == test_r.user) {
		// 		start = j;
		// 		while (user_id == test_r.user && j < ptr_train_data->size) {
		// 			j++;
		// 		}
		// 		end = j;
		// 		// TODO: make prediction with train_data[start:end] and test_r
		// 	}
		// 	else {
		// 		; // TODO: the user has no previous ratings. Return the average movie rating
		// 	}

		// 	start = j;

		// }



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
	string train_file_name = "../../data/mini_main.data";
	string test_file_name = "../../data/mini_prob.data";
	
	record_array train_data;
	train_data.load(train_file_name.c_str());
	cout << "finish loading " << train_file_name << endl;


	basic_rbm rbm;

	rbm.ptr_train_data = &train_data;
	rbm.fit(train_data);


	record_array test_data;
	test_data.load(test_file_name.c_str());
	rbm.ptr_test_data = &test_data;
	// rbm.predict_list();
}





#endif