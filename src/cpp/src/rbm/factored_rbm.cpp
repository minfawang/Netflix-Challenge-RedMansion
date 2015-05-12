#include <armadillo>
#include <iostream>
#include <omp.h>

#include "../types.hpp"


#ifndef __RBM_ESTIMATORS
#define __RBM_ESTIMATORS


#define NUM_THREADS 8
#define BATCH_SIZE (NUM_THREADS * 20)



using namespace arma;


bool isFuckedUp(double num) {
	return isnan(num) || isinf(num);
}


double sigma(double num) {
	return 1.0 / (1 + exp(-num));
}


class basic_rbm : public estimator_base {
public:

	cube W; // M * F * K
	cube A;
	mat B;
	mat BV; // K * M
	vec BH; // F
	// mat BH; // K * F

	unsigned int C;
	unsigned int N;
	unsigned int M;
	unsigned int K;
	unsigned int F;
	unsigned int CD_K;
	double lrate; // learning rate


	record_array *ptr_test_data;
	record_array *ptr_train_data;




	basic_rbm() {
		K = 5;
		F = 100;
		C = 30;
		M = 17770 / 10 + 1; // TODO: change M to be total number of movies
		N = 458293 / 10;

		W = randu<cube>(K, F, M) / 8.0;
		A = randu<cube>(K, C, M) / 8.0;
		B = randu<mat>(C, F) / 8.0;
		BV = randu<mat>(K, M) / 8.0;
		BH = randu<vec>(F) / 8.0;


		CD_K = 1;
		lrate = 0.05 / BATCH_SIZE;


	}

	virtual bool save(const char * file_name) {
		return true;
	}

	virtual bool load(const char * file_name) {
		return true;
	}


	virtual void fit(const record_array & train_data, unsigned int n_iter = 1, bool countinue_fit = false) {


		// training stage
		for (int iter_num = 0; iter_num < n_iter; iter_num++) {
			// customize CD_K based on the number of iteration
			if (iter_num < 15)
				CD_K = 1;
			else if (iter_num < 25)
				CD_K = 3;
			else if (iter_num < 35)
				CD_K = 5;
			else
				CD_K = 9;
			


			// TEST CODE
			vector<float> results = predict_list(*ptr_test_data);
			cout << "RMSE: " << RMSE(*ptr_test_data, results) << endl;
			
			cout << "working on iteration " << iter_num << "..." << endl;

			unsigned int user_id = train_data.data[0].user;
			unsigned int start = 0;
			unsigned int end = 0;



			int starts[BATCH_SIZE];
			int ends[BATCH_SIZE];
			int users[BATCH_SIZE];
			int thread_id = 0;
			starts[0] = 0;

			for (int i = 0; i < train_data.size; i++) {
				record r = train_data.data[i];
				if ((user_id != r.user) || i == train_data.size-1) {
					ends[thread_id] = (i == train_data.size-1) ? (i + 1) : i;
					users[thread_id] = user_id;

					user_id = r.user;
					thread_id++;

					// process a batch
					if (thread_id == (BATCH_SIZE) || i == train_data.size-1) {
#pragma omp parallel for num_threads(NUM_THREADS)
						for (int t = 0; t < thread_id; t++) {
							train(train_data.data+starts[t], users[t], ends[t]-starts[t], CD_K);
						}

						// update W
#pragma omp parallel for num_threads(NUM_THREADS)
						for (int iw = 0; iw < M; iw++) {
							W.slice(iw) = A.slice(iw) * B;
						}

						thread_id = 0;
					}
					starts[thread_id] = i;
				}
			}


			// store predicted data to file
			ofstream out_file;
		    out_file.open("test_coeff.txt");

		    // store W to file
		    for (int i = 0; i < M; i++) {
		    	for (int j = 0; j < F; j++) {
		    		out_file << W(0, j, i) << " ";
		    	}
		    	out_file << endl;
		    }

		    // store BV to file
		    for (int i = 0; i < M; i++) {
		    	for (int k = 0; k < K; k++) {
		    		out_file << BV(k, i) << " ";
		    	}
		    	out_file << endl;
		    }
		    
		    // store BH to file
		    for (int j = 0; j < F; j++) {
		    	out_file << BH(j) << endl;
		    }
		    
		    out_file.close();
		}


		cout << "finish training!" << endl;
		cout << "train data size: " << ptr_train_data->size << endl;
		cout << "test data size: " << ptr_test_data->size << endl;


	}





	vector<float> predict_list(const record_array & rcd_array) {
		// predicting stage
		unsigned int j = 0;
		unsigned int train_start = 0;
		unsigned int train_end = 0;
		unsigned int test_start = 0;
		unsigned int test_end = 0;
		unsigned int train_user = ptr_train_data->data[0].user;
		unsigned int test_user = ptr_test_data->data[0].user;

		vec Hu = zeros<vec>(F);
		vec Vum(K);
		ivec scores = linspace<ivec>(1, 5, 5);

		vector<float>results;
		results.resize(rcd_array.size);



		for (int i = 0; i < ptr_test_data->size; i++) {

			record r_test = ptr_test_data->data[i];

			if ((test_user != r_test.user) || i == ptr_test_data->size -1) {
				
				// make prediction of test_user for movies in the test set
				test_end = (i == ptr_test_data->size-1) ? (i + 1) : i;
				
				int u_size = test_end - test_start;

				// find train_start and train_end
				// record r_train = ptr_train_data->data[j];


				while (j < ptr_train_data->size) {
					record r_train = ptr_train_data->data[j];

					if (r_train.user < test_user) {
						train_start = j + 1;
					} else if (r_train.user > test_user) {
						break;
					}

					j++;
				}

				train_end = j;

				if (ptr_train_data->data[j-1].user == test_user) {

					// positive phase to compute Hu
					Hu = BH;
					for (int f = 0; f < F; f++) {
						for (int u = train_start; u < train_end; u++) {
							
							record r_train = ptr_train_data->data[u];
							unsigned int k = int(r_train.score) - 1;
							
							double w = W(k, f, r_train.movie);
							Hu(f) += w;
						}
					}
					Hu = 1.0 / (1 + exp(-Hu));


					// negative phase to predict score
					for (int u = test_start; u < test_end; u++) {
						record r_test = ptr_test_data->data[u];
						Vum = normalise( exp(BV.col(r_test.movie) + W.slice(r_test.movie) * Hu), 1);
						results[u] = dot(Vum, scores);
					}


				} else {
					// TODO: predict all movies to be the averaged movie rating
					double predict_score;
					for (int u = test_start; u < test_end; u++) {
						predict_score = 3.6;
						results[u] = predict_score;
					}
				}

				train_start = j;


				test_start = i;
				test_user = r_test.user;
			}
		}


	    return results;

	}



	virtual float predict(const record & rcd) {

		return 0.0;
	}




	void train(const record *data, unsigned int user_id, unsigned int size, int CD_K) {
		// initialization
		mat V0 = zeros<mat>(K, size);
		mat Vt = zeros<mat>(K, size);
		vec H0 = zeros<vec>(F);
		vec Ht = zeros<vec>(F);

		vector<int> ims(size);
		cube W_user(K, F, size);



		// set up V0 and Vt based on the input data.
		for (int i = 0; i < size; i++) {
			record r = data[i];
			V0(int(r.score)-1, i) = 1; // score - 1 is the index
			Vt(int(r.score)-1, i) = 1;

			ims[i] = r.movie;
			W_user.slice(i) = A.slice(r.movie) * B;
		}

		/*
		/////////////////// set up H0 by V -> H //////////////////
		H0(j) = sigma( BH(j) + sum_ik ( W(k, j, r.movie) * V0(k, i) ))
		*/

		H0 = BH;
		for (int i = 0; i < size; i++) {
			H0 += W_user.slice(i).t() * V0.col(i);
		}
		H0 = 1.0 / (1 + exp(-H0));
		


		/////////////////// Do the contrastive divergence ///////////////////
		for (int n = 0; n < CD_K; n++) {

			////////////// positive phase: V -> H /////////
			Ht = BH;
			for (int i = 0; i < size; i ++) {
				// Ht += W.slice(ims[i]).t() * Vt.col(i);
				Ht += W_user.slice(i).t() * Vt.col(i);
			}
			Ht = 1.0 / (1 + exp(-Ht));
			

			// negative phase: H -> V
			for (int i = 0; i < size; i++) {
				// Vt.col(i) = exp(BV.col(ims[i]) + W.slice(ims[i]) * Ht);
				Vt.col(i) = exp(BV.col(ims[i]) + W_user.slice(i) * Ht);
			}

			// Normalize Vt -> sum_k (Vt(k, i)) = 1
			Vt = normalise(Vt, 1);

		}

		// // update W
		// for (int i = 0; i < size; i++) {
		// 	W.slice(data[i].movie) += lrate * (V0.col(i) * H0.t() - Vt.col(i) * Ht.t());
		// }



		// update BH
		BH += lrate * (H0 - Ht);



		// update B
		// update BV
		// update A
		mat B_old = B;
		for (int i = 0; i < size; i++) {
			mat HV_diff = (V0.col(i) * H0.t() - Vt.col(i) * Ht.t());
			BV.col(ims[i]) += lrate * (V0.col(i) - Vt.col(i));
			B += lrate * A.slice(ims[i]).t() * HV_diff;
			A.slice(ims[i]) += lrate * HV_diff * B_old.t();
		}


	}


};


int main () {
	string train_file_name = "../../../data/mini_main.data";
	string test_file_name = "../../../data/mini_prob.data";
	// string train_file_name = "../../../data/main_data.data";
	// string test_file_name = "../../../data/prob_data.data";
	
	record_array train_data;
	train_data.load(train_file_name.c_str());
	cout << "finish loading " << train_file_name << endl;


	basic_rbm rbm;

	rbm.ptr_train_data = &train_data;


	record_array test_data;
	test_data.load(test_file_name.c_str());
	cout << "finish loading " << test_file_name << endl;
	rbm.ptr_test_data = &test_data;


	unsigned int iter_num = 20;
	rbm.fit(train_data, iter_num);

	vector<float>results = rbm.predict_list(test_data);
	cout << "RMSE: " << RMSE(test_data, results) << endl;


	// store results
	ofstream rbm_out_file;
	rbm_out_file.open("test_rbm_out.txt");
	for (int i = 0; i < test_data.size; i++) {
		rbm_out_file << results[i] << endl;
	}
	rbm_out_file.close();

}





#endif