#include "../includes.hpp"
#include <armadillo>
#include <iomanip>
#include <omp.h>

#ifndef __MF_ESTIMATORS
#define __MF_ESTIMATORS

using namespace arma;

class basic_mf : public estimator_base {
public:
	const int model_id = 10001;

	bool initialized;

	record_array * ptr_test_data;

	mat U;
	mat V;
	vec A;
	vec B;
	double mu;

	double lambda;
	double learning_rate;
	double learning_rate_per_record;
	double learning_rate_mul;

	unsigned int K;

	unsigned int n_iter;


	basic_mf() {
		K = 100;

		initialized = false;
		ptr_test_data = NULL;
		lambda = 0.05;
		learning_rate = 0.0005;
		learning_rate_mul = 1;
		
	}

	virtual bool save(const char * file_name) {
		ofstream output_file(file_name, ios::binary | ios::out);
		output_file.write((const char *)&model_id, sizeof(int));
		output_file.write((const char *)&K, sizeof(int));
		U.save(output_file);
		V.save(output_file);
		A.save(output_file);
		B.save(output_file);
		output_file << mu;
		return output_file.good();
	}

	virtual bool load(const char * file_name) {
		ifstream input_file(file_name, ios::binary | ios::in);
		int id;
		input_file.read((char *)&id, sizeof(int));
		input_file.read((char *)&K, sizeof(int));
		if (id != model_id) {
			cout << "FATAL: Loading Error" << endl;
			cout << model_id << ":" << id << endl;
			return false;
		}
		U.load(input_file);
		V.load(input_file);
		A.load(input_file);
		B.load(input_file);
		input_file >> mu;
		initialized = true;
		return input_file.good();
	}

	virtual float predict(const record & rcd) const{
		unsigned int i = rcd.user - 1, j = rcd.movie - 1;
		double result = as_scalar(U.col(i).t() * V.col(j)) + A(i) + B(j) + mu;
		return result;
	}

	void reshuffle(unsigned int *shuffle_idx, unsigned int n) {
		for (int i = n - 1; i > 0; i--) {
			int j = rand() % (i + 1);
			swap(shuffle_idx[i], shuffle_idx[j]);
		}
	}

	void update(const record & rcd) {
		unsigned int i = rcd.user - 1, j = rcd.movie - 1;
		double r_pFpX;
		double *Ui = U.colptr(i);
		double *Vj = V.colptr(j);

		double UiVj = fang_mul(Ui, Vj, K);

		r_pFpX = learning_rate_per_record * 2.0 * (rcd.score - mu - (UiVj + A(i) + B(j)));

		// U(:,i) = U(:,i) - rate * gUi; gUi = - pFpX * V(:,j);
		fang_add_mul(Ui, Vj, r_pFpX, K);

		// V(:,j) = V(:,j) - rate * gVj; gVj = - pFpX * U(:,i);
		fang_add_mul(Vj, Ui, r_pFpX, K);

		// A(:,i) = A(:,i) - rate * gAi; gAi = - pFpX;
		A(i) += r_pFpX;

		// B(:,j) = B(:,j) - rate * gBj; gBj = - pFpX;
		B(j) += r_pFpX;

	}

	virtual void fit(const record_array & train_data, unsigned int n_iter = 1, bool continue_fit=false) {
		try {
			unsigned int batch_size = 1000;
			unsigned int block_size = train_data.size / batch_size / 16;
			double shrink = 1 - lambda;
			unsigned int n_user = 0, n_movie = 0;
			unsigned int *shuffle_idx;
			unsigned int *shuffle_idx_batch;
			timer tmr;

			tmr.display_mode = 1;
			learning_rate_per_record = learning_rate;

			// Generate shuffle_idx
			cout << train_data.size << endl;

			shuffle_idx = new unsigned int[train_data.size / batch_size];
			for (int i = 0; i < train_data.size / batch_size; i++) {
				shuffle_idx[i] = i;
			}
			//shuffle_idx_batch = new unsigned int[batch_size];
			//for (int i = 0; i < batch_size; i++) {
			//	shuffle_idx_batch[i] = i;
			//}

			if (!continue_fit) {

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

			}

			for (int i_iter = 0; i_iter < n_iter; i_iter++) {

				tmr.tic();
				cout << "Iter\t" << i_iter << '\t';

				// Reshuffle first
				reshuffle(shuffle_idx, train_data.size / batch_size);

 #pragma omp parallel for num_threads(8)
				for (int i = 0; i < train_data.size / batch_size; i++) {
					unsigned int index_base = shuffle_idx[i] * batch_size;
					//reshuffle(shuffle_idx_batch, batch_size);

					for (int j = 0; j < batch_size; j++) {
						unsigned int index = index_base + j;

						// shuffle_idx_batch[j] do harm to the result
						if (index < train_data.size) {
							const record& rcd = train_data[index];
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
					cout << '\t' << RMSE(*ptr_test_data, result);
				}


				cout << '\t';
				tmr.toc();

				cout << "\t\t";
				cout << max(max(abs(U))) << '\t' << max(max(abs(V))) << '\t' << max(abs(A)) << '\t' << max(abs(B)) << endl;

				if (i_iter != n_iter - 1) {
				// Regularization
					U *= shrink;
					V *= shrink;
					A *= shrink;
					B *= shrink;
					learning_rate_per_record *= learning_rate_mul;
				}
			}
			delete[]shuffle_idx;
		}
		catch (std::bad_alloc & ba) {
			cout << "bad_alloc caught: " << ba.what() << endl;
			system("pause");
		}
	}

};


class alpha_mf : public estimator_base {
public:
	const int model_id = 10002;

	bool initialized;

	record_array * ptr_test_data;

	mat U;
	mat V;
	vec A;
	vec B;
	double mu;

	double lambda;
	double learning_rate;
	double learning_rate_per_record;
	double learning_rate_mul;

	unsigned int K;

	unsigned int n_iter;


	alpha_mf() {
		K = 20;

		initialized = false;
		ptr_test_data = NULL;
		lambda = 0.05;
		learning_rate = 0.0005;
		learning_rate_mul = 1;

	}

	virtual bool save(const char * file_name) {
		ofstream output_file(file_name, ios::binary | ios::out);
		output_file.write((const char *)&model_id, sizeof(int));
		output_file.write((const char *)&K, sizeof(int));
		U.save(output_file);
		V.save(output_file);
		A.save(output_file);
		B.save(output_file);
		output_file << mu;
		return output_file.good();
	}

	virtual bool load(const char * file_name) {
		ifstream input_file(file_name, ios::binary | ios::in);
		int id;
		input_file.read((char *)&id, sizeof(int));
		input_file.read((char *)&K, sizeof(int));
		if (id != model_id) {
			cout << "FATAL: Loading Error" << endl;
			cout << model_id << ":" << id << endl;
			return false;
		}
		U.load(input_file);
		V.load(input_file);
		A.load(input_file);
		B.load(input_file);
		input_file >> mu;
		initialized = true;
		return input_file.good();
	}

	virtual float predict(const record & rcd) const{
		unsigned int i = rcd.user - 1, j = rcd.movie - 1;
		double result = as_scalar(U.col(i).t() * V.col(j)) + A(i) + B(j) + mu;
		return result;
	}

	void reshuffle(unsigned int *shuffle_idx, unsigned int n) {
		for (int i = n - 1; i > 0; i--) {
			int j = rand() % (i + 1);
			swap(shuffle_idx[i], shuffle_idx[j]);
		}
	}

	void update(const record & rcd) {
		unsigned int i = rcd.user - 1, j = rcd.movie - 1;
		double r_pFpX;
		double *Ui = U.colptr(i);
		double *Vj = V.colptr(j);

		double UiVj = fang_mul(Ui, Vj, K);
		double data_mul = 0.2 * (2243 - rcd.date) / 2243 + 0.8;

		r_pFpX = data_mul * learning_rate_per_record * 2.0 * (rcd.score - mu - (UiVj + A(i) + B(j)));

		// U(:,i) = U(:,i) - rate * gUi; gUi = - pFpX * V(:,j);
		fang_add_mul(Ui, Vj, r_pFpX, K);

		// V(:,j) = V(:,j) - rate * gVj; gVj = - pFpX * U(:,i);
		fang_add_mul(Vj, Ui, r_pFpX, K);

		// A(:,i) = A(:,i) - rate * gAi; gAi = - pFpX;
		A(i) += r_pFpX;

		// B(:,j) = B(:,j) - rate * gBj; gBj = - pFpX;
		B(j) += r_pFpX;

	}

	virtual void fit(const record_array & train_data, unsigned int n_iter = 1, bool continue_fit = false) {
		try {
			unsigned int batch_size = 1000;
			unsigned int block_size = train_data.size / batch_size / 16;
			double shrink = 1 - lambda;
			unsigned int n_user = 0, n_movie = 0;
			unsigned int *shuffle_idx;
			unsigned int *shuffle_idx_batch;
			timer tmr;

			tmr.display_mode = 1;
			learning_rate_per_record = learning_rate;

			// Generate shuffle_idx
			cout << train_data.size << endl;

			shuffle_idx = new unsigned int[train_data.size / batch_size];
			for (int i = 0; i < train_data.size / batch_size; i++) {
				shuffle_idx[i] = i;
			}
			//shuffle_idx_batch = new unsigned int[batch_size];
			//for (int i = 0; i < batch_size; i++) {
			//	shuffle_idx_batch[i] = i;
			//}

			if (!continue_fit) {

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

			}

			for (int i_iter = 0; i_iter < n_iter; i_iter++) {

				tmr.tic();
				cout << "Iter\t" << i_iter << '\t';

				// Reshuffle first
				reshuffle(shuffle_idx, train_data.size / batch_size);

#pragma omp parallel for num_threads(8)
				for (int i = 0; i < train_data.size / batch_size; i++) {
					unsigned int index_base = shuffle_idx[i] * batch_size;
					//reshuffle(shuffle_idx_batch, batch_size);

					for (int j = 0; j < batch_size; j++) {
						unsigned int index = index_base + j;

						// shuffle_idx_batch[j] do harm to the result
						if (index < train_data.size) {
							const record& rcd = train_data[index];
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
					cout << '\t' << RMSE(*ptr_test_data, result);
				}


				cout << '\t';
				tmr.toc();

				cout << "\t\t";
				cout << max(max(abs(U))) << '\t' << max(max(abs(V))) << '\t' << max(abs(A)) << '\t' << max(abs(B)) << endl;

				if (i_iter != n_iter - 1) {
					// Regularization
					U *= shrink;
					V *= shrink;
					A *= shrink;
					B *= shrink;
					learning_rate_per_record *= learning_rate_mul;
				}
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
