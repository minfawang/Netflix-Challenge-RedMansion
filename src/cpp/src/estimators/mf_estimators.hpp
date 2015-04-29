#include "../includes.hpp"
#include <armadillo>
#include <iomanip>
#include <omp.h>
#include <unordered_map>
#include <tuple>
const double PI = 3.141592653589793238463;

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
	mat A_timebin;
	mat B_timebin;

	double mu;

	unsigned int K;
	unsigned int D_u;
	unsigned int D_i;

	double lambda;
	double learning_rate;
	double learning_rate_per_record;
	double learning_rate_mul;



	alpha_mf() {
		K = 20;
		D_u = 80;
		D_i = 40;

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

	unsigned int get_timebin(const unsigned int date, const unsigned int D) const{
		return (date - 1) * D / 2243;
	}

	virtual float predict(const record & rcd) const{
		unsigned int i = rcd.user - 1, j = rcd.movie - 1, 
					 d_i = get_timebin(rcd.date, D_u), d_j = get_timebin(rcd.date, D_i);
		double result = as_scalar(U.col(i).t() * V.col(j)) 
						+ A(i) + A_timebin(d_i, i)
						+ B(j) + B_timebin(d_j, j)
						+ mu;
		if (result > 5) {
			result = 5;
		}
		if (result < 1) {
			result = 1;
		}
		return result;
	}

	void reshuffle(unsigned int *shuffle_idx, unsigned int n) {
		for (int i = n - 1; i > 0; i--) {
			int j = rand() % (i + 1);
			swap(shuffle_idx[i], shuffle_idx[j]);
		}
	}


	void update(const record & rcd) {
		unsigned int i = rcd.user - 1, j = rcd.movie - 1, 
			         d_i = get_timebin(rcd.date, D_u), d_j = get_timebin(rcd.date, D_i);
		double r_pFpX;
		double *Ui = U.colptr(i);
		double *Vj = V.colptr(j);

		double UiVj = fang_mul(Ui, Vj, K);
		double data_mul = 1; // 0.2 * (2243 - rcd.date) / 2243 + 0.8;

		//learning rate * pFpX
		r_pFpX = data_mul * learning_rate_per_record * 2.0 * 
					(rcd.score - (UiVj + A(i) + A_timebin(d_i, i) + B(j) + B_timebin(d_j, j) + mu));

		// U(:,i) = U(:,i) - rate * gUi; gUi = - pFpX * V(:,j);
		fang_add_mul(Ui, Vj, r_pFpX, K);

		// V(:,j) = V(:,j) - rate * gVj; gVj = - pFpX * U(:,i);
		fang_add_mul(Vj, Ui, r_pFpX, K);

		// A(:,i) = A(:,i) - rate * gAi; gAi = - pFpX;
		A(i) += r_pFpX;
		A_timebin(d_i, i) += r_pFpX;

		// B(:,j) = B(:,j) - rate * gBj; gBj = - pFpX;
		B(j) += r_pFpX;
		B_timebin(d_j, j) += r_pFpX;

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
				A_timebin.set_size(D_u, n_user);
				B_timebin.set_size(D_i, n_movie);

				U.fill(fill::randu);
				V.fill(fill::randu);
				A.fill(fill::randu);
				B.fill(fill::randu);
				A_timebin.fill(fill::zeros);
				B_timebin.fill(fill::zeros);

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

				//cout << "\t\t";
				//cout << max(max(abs(U))) << ' ' 
				//	 << max(max(abs(V))) << ' ' 
				//	 << max(abs(A)) << ' ' 
				//	 << max(max(abs(A_timebin))) << ' '
				//	 << max(abs(B)) << ' '
				//	 << max(max(abs(B_timebin))) <<endl;

				if (i_iter != n_iter - 1) {
					// Regularization
					U *= shrink;
					V *= shrink;
					A *= shrink;
					A_timebin *= shrink;
					B *= shrink;
					B_timebin *= shrink;
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

class beta_mf : public estimator_base {
public:
	const int model_id = 10002;

	bool initialized;

	record_array * ptr_test_data;

	mat U;
	mat V;

	mat A;
	mat B;

	// (F_i, n_user)
	mat A_function_para;

	// (F_j, n_movie)
	mat B_function_para;

	// For each column:
	// Row 0: function type
	// Row 1: lambda
	// Row ...£º function parameter
	mat A_function_list;
	mat B_function_list;

	vec date_origin_user;
	vec date_origin_movie;

	double mu;

	unsigned int K;
	unsigned int F_i;
	unsigned int F_j;

	double lambda;
	double learning_rate;
	double learning_rate_per_record;
	double learning_rate_mul;

	unordered_map<pair<double, double>, double> pow_buffer;
	unordered_map<double, double> cos_buffer;
	unordered_map<double, double> sin_buffer;



	beta_mf() {
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


	unsigned int get_timebin(const unsigned int date, const unsigned int D) const{
		return (date - 1) * D / 2243;
	}

	double abspwr(double t, double beta) const{
		int sign = (t > 0) - (t < 0);
		return sign * pow(abs(t), beta);
	}

	double cos_buffered(double x){
		auto item = cos_buffer.find(x);
		
		if (item == cos_buffer.end()) {
			double result = cos(x);
			pair<double, double> pair_to_insert;
			pair_to_insert.first = x;
			pair_to_insert.second = result;
			cos_buffer[x];

		}
	}

	double eval_function(const double function_para[], double date) {
		unsigned int func = round(function_para[0]);
		switch (func)
		{
		case 0: return 1; break;
		case 1: return abspwr(date / 2243.0, function_para[2]); break;
		case 2: return cos_buffered(date / 2243.0 * 2 * PI * function_para[2]); break;
		case 3: return sin_buffered(date / 2243.0 * 2 * PI * function_para[2]); break;
		default: return 0;
		}
	}

	vec eval_functions(const mat& function_list, unsigned int date, double date_origin) {
		vec result;
		result.set_size(function_list.n_cols);
		for (unsigned int j = 0; j < function_list.n_cols; j++) {
			result[j] = eval_function(function_list.colptr(j), date - date_origin);
		}
		return result;
	}

	virtual float predict(const record & rcd) {
		unsigned int i = rcd.user - 1, j = rcd.movie - 1, d = rcd.date;
		vec A_func_val = eval_functions(A_function_list, d, date_origin_user[i]);
		vec B_func_val = eval_functions(B_function_list, d, date_origin_movie[j]);
		double result = mu + as_scalar(U.col(i).t() * V.col(j));
		result += dot(A_func_val, A.col(i));
		result += dot(B_func_val, B.col(j));

		if (result > 5) {
			result = 5;
		}
		if (result < 1) {
			result = 1;
		}

		//if (isnan(result)) {
		//	cout << endl;
		//	cout << d << endl;
		//	cout << date_origin_user[i] << ' ' << date_origin_movie[j] << endl;
		//	cout << A_func_val << A.col(i);
		//	cout << B_func_val << B.col(j);
		//	cout << mu << ' ' << as_scalar(U.col(i).t() * V.col(j));
		//	system("pause");
		//}
		return result;
	}

	void reshuffle(unsigned int *shuffle_idx, unsigned int n) {
		for (int i = n - 1; i > 0; i--) {
			int j = rand() % (i + 1);
			swap(shuffle_idx[i], shuffle_idx[j]);
		}
	}


	void update(const record & rcd) {
		unsigned int i = rcd.user - 1, j = rcd.movie - 1, d = rcd.date;

		double r_pFpX;

		double *Ui = U.colptr(i);
		double *Vj = V.colptr(j);

		double UiVj = fang_mul(Ui, Vj, K);

		double data_mul = 1; // 0.2 * (2243 - rcd.date) / 2243 + 0.8;

		vec A_func_val = eval_functions(A_function_list, d, date_origin_user[i]);
		vec B_func_val = eval_functions(B_function_list, d, date_origin_movie[j]);

		double result = mu + UiVj;
		result += dot(A_func_val, A.col(i));
		result += dot(B_func_val, B.col(j));

		//learning rate * pFpX
		r_pFpX = data_mul * learning_rate_per_record * 2.0 * (rcd.score - result);

		// U(:,i) = U(:,i) - rate * gUi; gUi = - pFpX * V(:,j);
		fang_add_mul(Ui, Vj, r_pFpX, K);

		// V(:,j) = V(:,j) - rate * gVj; gVj = - pFpX * U(:,i);
		fang_add_mul(Vj, Ui, r_pFpX, K);

		// A(:,i) = A(:,i) - rate * gAi; gAi = - pFpX;
		fang_add_mul(A.colptr(i), A_func_val.memptr(), r_pFpX, A_func_val.n_rows);

		// B(:,j) = B(:,j) - rate * gBj; gBj = - pFpX;
		fang_add_mul(B.colptr(j), B_func_val.memptr(), r_pFpX, B_func_val.n_rows);

	}

	void init(const record_array & train_data) {
		unsigned int n_user = 0, n_movie = 0;
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

		// Calculate t_origin
		date_origin_user = vec(n_user, fill::zeros);
		date_origin_movie = vec(n_movie, fill::zeros);

		vec user_count(n_user, fill::zeros);
		vec movie_count(n_movie, fill::zeros);
		for (unsigned int index = 0; index < train_data.size; index++) {
			unsigned int i = train_data[index].user - 1;
			unsigned int j = train_data[index].movie - 1;
			unsigned int d = train_data[index].date;
			user_count[i]++;
			movie_count[j]++;
			date_origin_user[i] += d;
			date_origin_movie[j] += d;
		}

		date_origin_user /= user_count;
		date_origin_movie /= movie_count;

		uvec idx;

		// Handle _count == 0
		idx = find_nonfinite(date_origin_user);
		for (unsigned int v : idx) {
			date_origin_user[v] = 2243 / 2.0;
		}

		idx = find_nonfinite(date_origin_movie);
		for (unsigned int v : idx) {
			date_origin_movie[v] = 2243 / 2.0;
		}


		// Reshape the matrix based on n_user and n_movie
		U.set_size(K, n_user);
		V.set_size(K, n_movie);

		// Writing the function list
		unsigned int maximum_n_para = 3;

		vector<vec> A_function_list_raw;
		vector<vec> B_function_list_raw;

		vec func(maximum_n_para);

		// For A

		// Constant
		func.zeros();
		func[0] = 0;
		func[1] = 0.05; // Lambda
		A_function_list_raw.push_back(func);

		// Beta  = 0.4
		func.zeros();
		func[0] = 1;
		func[1] = 0.05; // Lambda
		func[2] = 0.4;
		A_function_list_raw.push_back(func);

		// cos and sin
		for (unsigned int i = 1; i <= 16; i++) {
			// cos
			func.zeros();
			func[0] = 2;
			func[1] = 0.05; // Lambda
			func[2] = i;
			A_function_list_raw.push_back(func);

			// sin
			func[0] = 3;
			func[1] = 0.05; // Lambda
			func[2] = i;
			A_function_list_raw.push_back(func);
		}
		
		A_function_list.resize(maximum_n_para, A_function_list_raw.size());
		for (unsigned int j = 0; j < A_function_list_raw.size(); j++) {
			A_function_list.col(j) = A_function_list_raw[j];
		}


		// For B

		// Constant
		func.zeros();
		func[0] = 0;
		func[1] = 0.05; // Lambda
		B_function_list_raw.push_back(func);

		// Beta  = 0.4
		func.zeros();
		func[0] = 1;
		func[1] = 0.05; // Lambda
		func[2] = 0.4;
		B_function_list_raw.push_back(func);

		B_function_list.resize(maximum_n_para, B_function_list_raw.size());
		for (unsigned int j = 0; j < B_function_list_raw.size(); j++) {
			B_function_list.col(j) = B_function_list_raw[j];
		}
		

		A.resize(A_function_list_raw.size(), n_user);
		B.resize(B_function_list_raw.size(), n_movie);

		
		U.fill(fill::randu);
		V.fill(fill::randu);
		A.fill(fill::randu);
		B.fill(fill::randu);

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
				init(train_data);
			}

			// Recalculate all the shrinks
			vec A_shrink(A.n_rows);
			vec B_shrink(B.n_rows);

			for (unsigned int i = 0; i < A.n_rows; i++) {
				A_shrink[i] = 1 - A_function_list.col(i)[1];
			}

			for (unsigned int i = 0; i < B.n_rows; i++) {
				B_shrink[i] = 1 - B_function_list.col(i)[1];
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
				cout << max(max(abs(U))) << ' '
					<< max(max(abs(V))) << ' '
					<< max(max(abs(A))) << ' '
					<< max(max(abs(B))) << endl;

				if (i_iter != n_iter - 1) {
					// Regularization
					U *= shrink;
					V *= shrink;
					for (unsigned int j = 0; j < A.n_cols; j++) {
						A.col(j) %= A_shrink; // Element wise multiplication
					}
					for (unsigned int j = 0; j < B.n_cols; j++) {
						B.col(j) %= B_shrink; // Element wise multiplication
					}
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
