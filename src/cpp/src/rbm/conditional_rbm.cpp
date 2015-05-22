#include <armadillo>
#include <iostream>
#include <unordered_map>
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

bool file_exists(const char *fileName) {
	struct stat fileInfo;
	return stat(fileName, &fileInfo) == 0;
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
	record_array *ptr_qual_data;

	unordered_map<unsigned int, int*> train_map;
	unordered_map<unsigned int, int*> test_map;
	unordered_map<unsigned int, int*> qual_map;
	mat D;



	basic_rbm() {
		K = 5;

		F = 300;
		C = 30;
		M = 17770 / 1 + 1; // TODO: change M to be total number of movies
		N = 458293 / 1 + 1;

		A = randu<cube>(K, C, M) / 8.0;
		B = randu<mat>(C, F) / 8.0;
		BV = randu<mat>(K, M) / 8.0;
		BH = randu<vec>(F) / 8.0;

		D = randu<mat>(F, M) / 8.0;

		CD_K = 1;
		lrate = 0.05 / BATCH_SIZE;


	}

	virtual bool save(const char * file_name) {
		return true;
	}

	virtual bool load(const char * file_name) {
		return true;
	}


	void saveAllParameters(int iter_num) {
		ostringstream prefix;
		prefix << "K" << K << "_F" << F << "_C" << C << "_M" << M << "_N" << N << "_iter" << iter_num;

		string out_dir = "cond_starting_parameters/";

		string outName_A = out_dir + prefix.str() + "_A.cub";
		string outName_B = out_dir + prefix.str() + "_B.mat";
		string outName_BV = out_dir + prefix.str() + "_BV.mat";
		string outName_BH = out_dir + prefix.str() + "_BH.vec";
		string outName_D = out_dir + prefix.str() + "_D.mat";
		
		A.save(outName_A, arma_binary);
		B.save(outName_B, arma_binary);
		BV.save(outName_BV, arma_binary);
		BH.save(outName_BH, arma_binary);
		D.save(outName_D, arma_binary);

	}

	void loadAllParameters(int iter_num) {

		ostringstream prefix;
		prefix << "K" << K << "_F" << F << "_C" << C << "_M" << M << "_N" << N << "_iter" << iter_num;

		string out_dir = "cond_starting_parameters/";

		string outName_A = out_dir + prefix.str() + "_A.cub";
		string outName_B = out_dir + prefix.str() + "_B.mat";
		string outName_BV = out_dir + prefix.str() + "_BV.mat";
		string outName_BH = out_dir + prefix.str() + "_BH.vec";
		string outName_D = out_dir + prefix.str() + "_D.mat";
		
		A.load(outName_A, arma_binary);
		B.load(outName_B, arma_binary);
		BV.load(outName_BV, arma_binary);
		BH.load(outName_BH, arma_binary);
		D.load(outName_D, arma_binary);
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
			// cout << "predicting ... " << endl;
			// vector<float> results = predict_list(*ptr_test_data);
			// float prob_rmse = RMSE(*ptr_test_data, results);
			// cout << "RMSE: " << prob_rmse << endl;
			// if (prob_rmse < 0.93) {
			// 	predict_qual_results_to_file(*ptr_qual_data, prob_rmse, iter_num);
			// }
			if (iter_num >= 10) {
				cout << "predicting ..." << endl;
				vector<float> results = predict_array(*ptr_test_data, *ptr_qual_data, test_map, qual_map);
				float prob_rmse = RMSE(*ptr_test_data, results);
				cout << "RMSE: " << prob_rmse << endl;
				if (prob_rmse < 0.925) {
					write_prob_results_to_file(results, prob_rmse, iter_num);
					predict_qual_results_to_file(prob_rmse, iter_num);
				}				
			}


			
			cout << "working on iteration " << iter_num << "..." << endl;

			int starts[BATCH_SIZE];
			int ends[BATCH_SIZE];
			int users[BATCH_SIZE];
			int thread_id = 0;


			for (unsigned int user = 1; user < N; ++user) {
				int* ids = train_map[user];
				users[thread_id] = user;
				starts[thread_id] = ids[0];
				ends[thread_id] = ids[1];

				thread_id++;

				// process a batch
				if (thread_id == BATCH_SIZE || user == N-1) {
#pragma omp parallel for num_threads(NUM_THREADS)
					for (int t = 0; t < thread_id; t++) {
						train(train_data.data+starts[t], users[t], ends[t]-starts[t], CD_K);
					}

					thread_id = 0;
				}

			}

			if (iter_num >= 10 && iter_num % 4 == 0) {
				saveAllParameters(iter_num);
			}

		}


		cout << "finish training!" << endl;
		cout << "train data size: " << ptr_train_data->size << endl;
		cout << "test data size: " << ptr_test_data->size << endl;


	}


	void train(const record *data, unsigned int user_id, unsigned int size, int CD_K) {
		// initialization
		mat V0 = zeros<mat>(K, size);
		mat Vt = zeros<mat>(K, size);
		vec H0 = zeros<vec>(F);
		vec Ht = zeros<vec>(F);

		vector<int> ims(size);
		cube W_user(K, F, size);

		// TEST CODE
		// vec r = zeros<vec>(M);
		int* train_ids = train_map[user_id];
		int* test_ids;
		int* qual_ids;
		unordered_map<unsigned int, int*>::const_iterator test_ids_iter = test_map.find(user_id);
		unordered_map<unsigned int, int*>::const_iterator qual_ids_iter = qual_map.find(user_id);

		int num_movies = train_ids[1] - train_ids[0];
		if (test_ids_iter != test_map.end()) {
			test_ids = test_ids_iter->second;
			num_movies += test_ids[1] - test_ids[0];
		}
		if (qual_ids_iter != qual_map.end()) {
			qual_ids = qual_ids_iter->second;
			num_movies += qual_ids[1] - qual_ids[0];
		}


		uvec r(num_movies);
		int movie_idx = 0;

		for (int train_id = train_ids[0]; train_id < train_ids[1]; ++train_id) {
			r(movie_idx) = ptr_train_data->data[train_id].movie;
			movie_idx++;
		}

		if (test_ids_iter != test_map.end())
			for (int test_id = test_ids[0]; test_id < test_ids[1]; ++test_id) {
				r(movie_idx) = ptr_test_data->data[test_id].movie;
				movie_idx++;
			}

		if (qual_ids_iter != qual_map.end()) 
			for (int qual_id = qual_ids[0]; qual_id < qual_ids[1]; ++qual_id) {
				r(movie_idx) = ptr_qual_data->data[qual_id].movie;
				movie_idx++;
			}


		vec Du_sum = sum(D.cols(r), 1);



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
		H0 += Du_sum;
		H0 = 1.0 / (1 + exp(-H0));


		/////////////////// Do the contrastive divergence ///////////////////
		for (int n = 0; n < CD_K; n++) {

			////////////// positive phase: V -> H /////////
			Ht = BH;
			for (int i = 0; i < size; i ++) {
				// Ht += W.slice(ims[i]).t() * Vt.col(i);
				Ht += W_user.slice(i).t() * Vt.col(i);
			}
			Ht += Du_sum;
			Ht = 1.0 / (1 + exp(-Ht));
			

			// negative phase: H -> V
			for (int i = 0; i < size; i++) {
				// Vt.col(i) = exp(BV.col(ims[i]) + W.slice(ims[i]) * Ht);
				Vt.col(i) = exp(BV.col(ims[i]) + W_user.slice(i) * Ht);
			}

			// Normalize Vt -> sum_k (Vt(k, i)) = 1
			Vt = normalise(Vt, 1);

		}


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

		// update D
		D.each_col(r) += lrate * (H0 - Ht);
	}


	vector<float> predict_array(const record_array &rcd_array, const record_array &helper_array, unordered_map<unsigned int, int*> &predict_map, unordered_map<unsigned int, int*> &helper_map) {
		vector<float>results(rcd_array.size);
		int users[BATCH_SIZE];
		// int starts[BATCH_SIZE];
		// int ends[BATCH_SIZE];

		int thread_id = 0;
		int batch_id = 0;
		int* test_ids;
		for (int user_id = 1; user_id < N; user_id++) {
			unordered_map<unsigned int, int*>::const_iterator test_ids_iter = predict_map.find(user_id);
			if (test_ids_iter != predict_map.end()) {
				// test_ids = test_ids_iter->second;

				users[thread_id] = user_id;
				// starts[thread_id] = test_ids[0];
				// ends[thread_id] = test_ids[1];

				thread_id++;

				if (thread_id == BATCH_SIZE) {
#pragma omp parallel for num_threads(NUM_THREADS)
					for (int t = 0; t < thread_id; t++) {
						predict_user(users[t], rcd_array, helper_array, predict_map, helper_map, results);
					}

					thread_id = 0;
					batch_id++;
				}

			}
		}

		// user == N
		if (thread_id != 0) {
#pragma omp parallel for num_threads(NUM_THREADS)
			for (int t = 0; t < thread_id; t++) {
				predict_user(users[t], rcd_array, helper_array, predict_map, helper_map, results);
			}		
		}

		return results;
	}


	void predict_user(int user, const record_array &rcd_array, const record_array &helper_data, unordered_map<unsigned int, int*> &predict_map, unordered_map<unsigned int, int*> &helper_map, vector<float> &results) {

		vec Vum(K);
		ivec scores = linspace<ivec>(1, 5, 5);
		int train_start;
		int train_end;
		int test_start;
		int test_end;
		int* test_ids = predict_map[user];
		int* train_ids = train_map[user];
		int* helper_ids;

		// set up r
		
		test_start = test_ids[0];
		test_end = test_ids[1];

		train_start = train_ids[0];
		train_end = train_ids[1];

		int movie_idx = 0;
		int num_movies = test_end - test_start;
		num_movies += train_end - train_start;

		unordered_map<unsigned int, int*>::const_iterator helper_ids_iter = helper_map.find(user);
		if (helper_ids_iter != helper_map.end()) {
			helper_ids = helper_ids_iter->second;
			num_movies += helper_ids[1] - helper_ids[0];
		}


		uvec r(num_movies);

		for (int train_id = train_ids[0]; train_id < train_ids[1]; ++train_id) {
			r(movie_idx) = ptr_train_data->data[train_id].movie;
			movie_idx++;
		}

		for (int test_id = test_ids[0]; test_id < test_ids[1]; ++test_id) {
			r(movie_idx) = rcd_array.data[test_id].movie;
			movie_idx++;
		}

		if (helper_ids_iter != helper_map.end()) {
			for (int helper_id = helper_ids[0]; helper_id < helper_ids[1]; ++helper_id) {
				r(movie_idx) = helper_data.data[helper_id].movie;
				movie_idx++;
			}
		}

		vec Du_sum = sum(D.cols(r), 1);


		// positive phase to compute Hu
		vec Hu = BH;
		for (int f = 0; f < F; f++) {
			for (int u = train_start; u < train_end; u++) {
				
				record r_train = ptr_train_data->data[u];
				unsigned int k = int(r_train.score) - 1;
				
				// double w = W(k, f, r_train.movie);
				double w  = 0;
				for (int c = 0; c < C; c++) {
					w += A(k, c, r_train.movie) * B(c, f);	
				}
				Hu(f) += w;
			}
		}
		Hu += Du_sum;
		Hu = 1.0 / (1 + exp(-Hu));

		// negative phase to predict score
		for (int u = test_start; u < test_end; u++) {
			record r_test = rcd_array.data[u];
			Vum = normalise( exp(BV.col(r_test.movie) + A.slice(r_test.movie) * B * Hu), 1);

			results[u] = dot(Vum, scores);
		}
	}

	virtual float predict(const record & rcd) {
		return 0.0;
	}


	void predict_qual_results_to_file(const float prob_rmse, unsigned int iter_num) {
		cout << "predicting qual data ..." << endl;
		vector<float>results = predict_array(*ptr_qual_data, *ptr_test_data, qual_map, test_map);

		// store results
		string out_dir = "crbm_results/";
		string rbm_out_name_pre;
		ostringstream convert;
		convert << prob_rmse << "_crbm" << "_lrate" << this->lrate << "_F" << this->F << "_C" << this->C << "_iter" << iter_num;
		rbm_out_name_pre = out_dir + convert.str();
		string rbm_out_name = rbm_out_name_pre;

		for (int file_idx = 1; file_exists(rbm_out_name.c_str()); file_idx++) {
			rbm_out_name = rbm_out_name_pre + "_idx" + to_string(file_idx);
		}

		cout << "write to file: " << rbm_out_name << endl;

		ofstream rbm_out_file;
		rbm_out_file.open(rbm_out_name);
		for (int i = 0; i < ptr_qual_data->size; i++) {
			rbm_out_file << results[i] << endl;
		}
		rbm_out_file.close();
	}

	void write_prob_results_to_file(vector<float> results, const float prob_rmse, unsigned int iter_num) {

		// store results
		string out_dir = "crbm_results/";
		string rbm_out_name_pre;
		ostringstream convert;
		convert << "prob" << prob_rmse << "_crbm" << "_lrate" << this->lrate << "_F" << this->F << "_C" << this->C << "_iter" << iter_num;
		rbm_out_name_pre = out_dir + convert.str();
		string rbm_out_name = rbm_out_name_pre;

		for (int file_idx = 1; file_exists(rbm_out_name.c_str()); file_idx++) {
			rbm_out_name = rbm_out_name_pre + "_idx" + to_string(file_idx);
		}

		cout << "write to file: " << rbm_out_name << endl;

		ofstream rbm_out_file;
		rbm_out_file.open(rbm_out_name);
		for (int i = 0; i < ptr_test_data->size; i++) {
			rbm_out_file << results[i] << endl;
		}
		rbm_out_file.close();
	}

};



unordered_map<unsigned int, int*> make_pre_map(const record_array &record_data) {
	unordered_map<unsigned int, int*> record_map;

	unsigned int cur_user = record_data.data[0].user;
	int cur_start = 0;
	int cur_end = 1;
	int* user_ids;
	for (int i = 0; i < record_data.size; i++) {
		record this_data = record_data.data[i];
		if (this_data.user != cur_user) {
			cur_end = i;
			

			user_ids = new int[2];
			user_ids[0] = cur_start;
			user_ids[1] = cur_end;
			record_map[cur_user] = user_ids;
			
			cur_user = this_data.user;
			cur_start = i;
		}
	}
	user_ids = new int[2];
	user_ids[0] = cur_start;
	user_ids[1] = record_data.size;
	record_map[cur_user] = user_ids;

	cout << "number of users = " << record_map.size() << endl;

	return record_map;
}





int main () {


	unsigned int ITER_NUM = 60;
	

	// string train_file_name = "../../../data/mini_main.data";
	// string test_file_name = "../../../data/mini_prob.data";
	// string qual_file_name = "../../../data/mini_prob.data"; // TODO: Change this name!!!
	string train_file_name = "../../../data/main_data.data";
	string test_file_name = "../../../data/prob_data.data";
	string qual_file_name = "../../../data/qual_data.data";
	
	record_array train_data;
	record_array test_data;
	record_array qual_data;
	train_data.load(train_file_name.c_str());
	cout << "finish loading " << train_file_name << endl;
	test_data.load(test_file_name.c_str());
	cout << "finish loading " << test_file_name << endl;
	qual_data.load(qual_file_name.c_str());
	cout << "finish loading " << qual_file_name << endl;


	basic_rbm rbm;
	rbm.ptr_train_data = &train_data;
	rbm.ptr_test_data = &test_data;
	rbm.ptr_qual_data = &qual_data;

	rbm.train_map = make_pre_map(train_data);
	rbm.test_map = make_pre_map(test_data);
	rbm.qual_map = make_pre_map(qual_data);


	rbm.fit(train_data, ITER_NUM);


	// TODO Rewrite calling for predict array
	vector<float> results = rbm.predict_array(test_data, qual_data, rbm.test_map, rbm.qual_map);
	float prob_rmse = RMSE(test_data, results);
	cout << "RMSE: " << prob_rmse << endl;

	if (prob_rmse < 0.925) {
		rbm.predict_qual_results_to_file(prob_rmse, ITER_NUM);
	}



}





#endif









	// vector<float> predict_list(const record_array & rcd_array) {
	// 	// predicting stage
	// 	unsigned int j = 0;
	// 	unsigned int train_start = 0;
	// 	unsigned int train_end = 0;
	// 	unsigned int test_start = 0;
	// 	unsigned int test_end = 0;
	// 	unsigned int train_user = ptr_train_data->data[0].user;
	// 	unsigned int test_user = rcd_array.data[0].user;

	// 	vec Hu = zeros<vec>(F);
	// 	vec Vum(K);
	// 	ivec scores = linspace<ivec>(1, 5, 5);

	// 	vector<float>results;
	// 	results.resize(rcd_array.size);



	// 	for (int i = 0; i < rcd_array.size; i++) {

	// 		record r_test = rcd_array.data[i];

	// 		if ((test_user != r_test.user) || i == rcd_array.size -1) {
				
	// 			// make prediction of test_user for movies in the test set
	// 			test_end = (i == rcd_array.size-1) ? (i + 1) : i;
				
	// 			int u_size = test_end - test_start;

	// 			// find train_start and train_end
	// 			// record r_train = ptr_train_data->data[j];


	// 			while (j < ptr_train_data->size) {
	// 				record r_train = ptr_train_data->data[j];

	// 				if (r_train.user < test_user) {
	// 					train_start = j + 1;
	// 				} else if (r_train.user > test_user) {
	// 					break;
	// 				}

	// 				j++;
	// 			}

	// 			train_end = j;

	// 			if (ptr_train_data->data[j-1].user == test_user) {

	// 				// positive phase to compute Hu
	// 				Hu = BH;
	// 				for (int f = 0; f < F; f++) {
	// 					for (int u = train_start; u < train_end; u++) {
							
	// 						record r_train = ptr_train_data->data[u];
	// 						unsigned int k = int(r_train.score) - 1;
							
	// 						// double w = W(k, f, r_train.movie);
	// 						double w  = 0;
	// 						for (int c = 0; c < C; c++) {
	// 							w += A(k, c, r_train.movie) * B(c, f);	
	// 						}
	// 						Hu(f) += w;
	// 					}
	// 				}
	// 				Hu = 1.0 / (1 + exp(-Hu));


	// 				// negative phase to predict score
	// 				for (int u = test_start; u < test_end; u++) {
	// 					record r_test = rcd_array.data[u];
	// 					// Vum = normalise( exp(BV.col(r_test.movie) + W.slice(r_test.movie) * Hu), 1);
	// 					Vum = normalise( exp(BV.col(r_test.movie) + A.slice(r_test.movie) * B * Hu), 1);
	// 					results[u] = dot(Vum, scores);
	// 				}


	// 			} else {
	// 				// TODO: predict all movies to be the averaged movie rating
	// 				double predict_score;
	// 				for (int u = test_start; u < test_end; u++) {
	// 					predict_score = 3.6;
	// 					results[u] = predict_score;
	// 				}
	// 			}

	// 			train_start = j;


	// 			test_start = i;
	// 			test_user = r_test.user;
	// 		}
	// 	}


	//     return results;

	// }



	// virtual float predict(const record & rcd) {

	// 	return 0.0;
	// }



	// void predict_qual_results_to_file(const record_array &qual_data, const float prob_rmse, unsigned int iter_num) {
	// 	cout << "predicting qual data ..." << endl;
	// 	vector<float>results = this->predict_list(qual_data);
	// 	// store results
	// 	string out_dir = "frbm_results/";
	// 	string rbm_out_name_pre;
	// 	ostringstream convert;
	// 	convert << prob_rmse << "_lrate" << this->lrate << "_F" << this->F << "_C" << this->C << "_iter" << iter_num;
	// 	rbm_out_name_pre = out_dir + convert.str();
	// 	string rbm_out_name = rbm_out_name_pre;

	// 	for (int file_idx = 1; file_exists(rbm_out_name.c_str()); file_idx++) {
	// 		rbm_out_name = rbm_out_name_pre + "_idx" + to_string(file_idx);
	// 	}

	// 	cout << "write to file: " << rbm_out_name << endl;

	// 	ofstream rbm_out_file;
	// 	rbm_out_file.open(rbm_out_name);
	// 	for (int i = 0; i < qual_data.size; i++) {
	// 		rbm_out_file << results[i] << endl;
	// 	}
	// 	rbm_out_file.close();
	// }