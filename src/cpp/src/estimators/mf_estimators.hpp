#include "../includes.hpp"
#include <armadillo>
#include <iomanip>
#include <omp.h>
#include <unordered_map>
#include <tuple>
#include <mutex>

const int MAX_DATE = 2243;
const double PI = 3.141592653589793238463;
const int N_THREADS = 8;

//#define _TEST_NAN
#define _USE_Y 0

#ifndef __MF_ESTIMATORS
#define __MF_ESTIMATORS

using namespace arma;

class function_table_generator {
private:
	double abspwr(int t, float beta) const{
		int sign = (t > 0) - (t < 0);
		return sign * pow(abs(1.0 * t / MAX_DATE), beta);
	}
	double sinw(int t, float w) const {
		return sin(1.0 * t / MAX_DATE * PI * w);
	}
	double cosw(int t, float w) const {
		return cos(1.0 * t / MAX_DATE * PI * w);
	}
public:
	rowvec const_table() const {
		rowvec result(2 * MAX_DATE + 1);
		result.fill(fill::ones);
		return result;
	}

	rowvec abspwr_table(float beta) const {
		rowvec result(2 * MAX_DATE + 1);
		for (int i = -MAX_DATE; i <= MAX_DATE; i++) {
			result[i + MAX_DATE] = abspwr(i, beta);
		}
		return result;
	}

	rowvec sinw_table(float w) const {
		rowvec result(2 * MAX_DATE + 1);
		for (int i = -MAX_DATE; i <= MAX_DATE; i++) {
			result[i + MAX_DATE] = sinw(i, w);
		}
		return result;
	}

	rowvec cosw_table(float w) const {
		rowvec result(2 * MAX_DATE + 1);
		for (int i = -MAX_DATE; i <= MAX_DATE; i++) {
			result[i + MAX_DATE] = cosw(i, w);
		}
		return result;
	}
};

double eval_function(const mat &function_table, int t) {
	return function_table[t + MAX_DATE];
}

double* eval_functions(mat &function_table, int t) {
	return function_table.colptr(t + MAX_DATE);
}

int get_timebin(const int date, const int D) {
	return (date - 1) * D / MAX_DATE;
}

class gamma_mf : public estimator_base {
public:
	const int model_id = 10002;

	bool initialized;

	record_array * ptr_test_data;
	record_array * ptr_qual_data;

    char * learning_rate_file;
    char * lambda_file;

	mat U0;
    double U0_learning_rate;
    double U0_lambda;

	mat U1;
    double U1_learning_rate;
    double U1_lambda;

	rowvec U1_function_table;

	mat V;
    double V_learning_rate;
    double V_lambda;

	mat Y;
    double Y_learning_rate;
    double Y_lambda;

	// (F_i, n_user)
	mat A;
	// (F_j, n_movie)
	mat B;
	
    vec A_learning_rate;

	vec A_lambda;	

    vec B_learning_rate;

	vec B_lambda;

	// (F_i, t + MaxDate)
	mat A_function_table;

	// (F_j, t + MaxDate)
	mat B_function_table;

    // (D_u, n_user)
	mat A_timebin;

    double A_timebin_learning_rate;
    double A_timebin_lambda;

    // (D_i, n_movie)
	mat B_timebin;

    double B_timebin_learning_rate;
    double B_timebin_lambda;

    // (D_u, n_user)
    mat C_timebin;
    double C_timebin_learning_rate;
    double C_timebin_lambda;

	uvec date_origin_user;
	uvec date_origin_movie;

    // (D_if, n_movie)
    //mat B_if;




	struct {
		vec U_col;
		vec A_function_val;
		vec B_function_val;
		vec Yi;
	} update_temp_var_thread[N_THREADS];

	vector<vector<int>> vote_list;

	double mu;
	double scale;

	int K;
	int F_i;
	int F_j;

	int D_u;
	int D_i;
    //int D_if;

	double lambda_factor;


	double learning_rate;
	double learning_rate_per_record;

	double learning_rate_mul;
	double learning_rate_min;

    double rmse_sum;
    double rmse_count;
	gamma_mf() {
		K = 20;
        scale = 1;
		D_u = 20;
		D_i = 20;
        //D_if = 10;

		initialized = false;
		ptr_test_data = NULL;
        ptr_qual_data = NULL;

		lambda_factor = 1;
		U0_lambda = 0.0001;
		U1_lambda = 0.1;
		V_lambda = 0.001;
		Y_lambda = 0.0001;


		// learning_rate = 0.002;
		learning_rate = 0.0015;

		learning_rate_mul = 0.9;
		learning_rate_min = 0;
	}

	virtual bool save(const char * file_name) {
		ofstream output_file(file_name, ios::binary | ios::out);
		output_file.write((const char *)&model_id, sizeof(int));
		output_file.write((const char *)&K, sizeof(int));
		U0.save(output_file);
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
		U0.load(input_file);
		V.load(input_file);
		A.load(input_file);
		B.load(input_file);
		input_file >> mu;
		initialized = true;
		return input_file.good();
	}

    void load_learning_rate(char *file_name) {
        ifstream input(file_name);

        input >> U0_learning_rate;
        input >> U1_learning_rate;
        input >> V_learning_rate;
        input >> Y_learning_rate;

        A_learning_rate.resize(A_function_table.n_rows);
        for (int i = 0; i < A_function_table.n_rows; i++) {
            input >> A_learning_rate[i];
        }

        B_learning_rate.resize(B_function_table.n_rows);
        for (int i = 0; i < B_function_table.n_rows; i++) {
            input >> B_learning_rate[i];
        }

        input >> A_timebin_learning_rate;
        input >> B_timebin_learning_rate;
        input >> C_timebin_learning_rate;
    }

    void load_lambda(char *file_name) {
        ifstream input(file_name);
        input >> U0_lambda;
        input >> U1_lambda;
        input >> V_lambda;
        input >> Y_lambda;

        A_lambda.resize(A_function_table.n_rows);
        for (int i = 0; i < A_function_table.n_rows; i++) {
            input >> A_lambda[i];
        }

        B_lambda.resize(B_function_table.n_rows);
        for (int i = 0; i < B_function_table.n_rows; i++) {
            input >> B_lambda[i];
        }

        input >> A_timebin_lambda;
        input >> B_timebin_lambda;
        input >> C_timebin_lambda;

    }

	void get_Y(vec& rtn, int i) {
		rtn.resize(K);
		rtn.fill(fill::zeros);
		vector<int>& user_vote_list = vote_list[i];
		for (int j : user_vote_list) {
			rtn += Y.unsafe_col(j);
		}
		if (user_vote_list.size() > 0) {
			rtn /= user_vote_list.size();
		}
	}

	void update_Y(int i, double r_pFpX, double *Vj, int K) {
		vector<int>& user_vote_list = vote_list[i];
		r_pFpX /= user_vote_list.size();
		for (int j : user_vote_list) {
			fang_add_mul(Y.colptr(j), Vj, r_pFpX, K);
		}
	}

	virtual float predict(const record & rcd) {
		int i = rcd.user - 1, j = rcd.movie - 1, d = rcd.date;
		int d_i = get_timebin(rcd.date, D_u), d_j = get_timebin(rcd.date, D_i);

		double* A_func_val = eval_functions(A_function_table, d - date_origin_user[i]);
		double* B_func_val = eval_functions(B_function_table, d - date_origin_movie[j]);

        double c = C_timebin(d_i, i);

		vec U_col(K);
	    fang_add_mul_rtn(U_col, U0.colptr(i), U1.colptr(i), eval_function(U1_function_table, d - date_origin_user[i]), U0.n_rows);

		double result = mu + dot(U_col, V.unsafe_col(j));

		result += fang_mul(A_func_val, A.colptr(i), A.n_rows);
		result += (c + 1) * fang_mul(B_func_val, B.colptr(j), B.n_rows);

		result += A_timebin(d_i, i);
		result += (c + 1) * B_timebin(d_j, j);

#if _USE_Y
		vec Y_i;
		get_Y(Y_i, i);
		result += dot(V.unsafe_col(j), Y_i);
#endif


		if (result > 5) {
			result = 5;
		}
		if (result < 1) {
			result = 1;
		}

		return result;
	}

	void reshuffle(int *shuffle_idx, int n) {
		for (int i = n - 1; i > 0; i--) {
			int j = rand() % (i + 1);
			swap(shuffle_idx[i], shuffle_idx[j]);
		}
	}


	void update(const record & rcd, int tid) {
		int i = rcd.user - 1, j = rcd.movie - 1, d = rcd.date;
		int d_i = get_timebin(rcd.date, D_u), d_j = get_timebin(rcd.date, D_i);

		double r_pFpX;

		double *U0i = U0.colptr(i);
		double *U1i = U1.colptr(i);
		double u = eval_function(U1_function_table, d - date_origin_user[i]);

		vec &U_col = update_temp_var_thread[tid].U_col;
		double *A_func_val;
		double *B_func_val;

        double c = C_timebin(d_i, i);

		vec &Yi = update_temp_var_thread[tid].Yi;

		fang_add_mul_rtn(U_col, U0i, U1i, u, U0.n_rows);

		double *Vj = V.colptr(j);
		double UiVj = fang_mul(U_col.memptr(), V.colptr(j), K);

		double data_mul = 1; // 0.2 * (2243 - rcd.date) / 2243 + 0.8;

		A_func_val = eval_functions(A_function_table, d - date_origin_user[i]);
		B_func_val = eval_functions(B_function_table, d - date_origin_movie[j]);

        double a1 = fang_mul(A_func_val, A.colptr(i), A.n_rows);
        double a2 = A_timebin(d_i, i);
        double b1 = fang_mul(B_func_val, B.colptr(j), B.n_rows);
        double b2 = B_timebin(d_j, j);

		double result = mu + UiVj;
        result += a1 + a2;
        result += (c + 1) * (b1 + b2);

#if _USE_Y
		get_Y(Yi, i);
		result += dot(V.unsafe_col(j), Yi);
#endif
        result = rcd.score - result;

        rmse_sum += result * result;
        rmse_count += 1;
		//learning rate * pFpX
		r_pFpX = data_mul * 2.0 * result;

		// U(:,i) = U(:,i) - rate * gUi; gUi = - pFpX * V(:,j);
		fang_add_mul(U0i, Vj, r_pFpX * U0_learning_rate * scale, K);
		fang_add_mul(U1i, Vj, r_pFpX * U1_learning_rate * scale, K);

		// V(:,j) = V(:,j) - rate * gVj; gVj = - pFpX * U(:,i);
		fang_add_mul(Vj, U_col.memptr(), r_pFpX * V_learning_rate * scale, K);

		// A(:,i) = A(:,i) - rate * gAi; gAi = - pFpX;
        fang_add_mul2(A.colptr(i), A_func_val, A_learning_rate.memptr(), r_pFpX * scale, A.n_rows);

		// B(:,j) = B(:,j) - rate * gBj; gBj = - pFpX;
        fang_add_mul2(B.colptr(j), B_func_val, B_learning_rate.memptr(), r_pFpX * (c + 1) * scale, B.n_rows);

        A_timebin(d_i, i) += r_pFpX * A_timebin_learning_rate * scale;
        B_timebin(d_j, j) += r_pFpX * (c + 1) * B_timebin_learning_rate * scale;
        C_timebin(d_i, i) += r_pFpX * (b1 + b2) * C_timebin_learning_rate * scale;

#if _USE_Y
		update_Y(i, r_pFpX * Y_learning_rate, Vj, K);
#endif
	}

    void regular() {
        vec A_shrink(A.n_rows);
        vec B_shrink(B.n_rows);


        double regu_pwr = lambda_factor;
        // Recalculate all the shrinks
        for (int i = 0; i < A.n_rows; i++) {
            A_shrink[i] = pow(1 - A_lambda[i] * A_learning_rate[i] * scale, regu_pwr);
        }

        for (int i = 0; i < B.n_rows; i++) {
            B_shrink[i] = pow(1 - B_lambda[i] * B_learning_rate[i] * scale, regu_pwr);
        }

        // Regularization

        U0 *= pow(1 - U0_lambda * U0_learning_rate * scale, regu_pwr);
        U1 *= pow(1 - U1_lambda * U1_learning_rate * scale, regu_pwr);
        V *= pow(1 - V_lambda * V_learning_rate * scale, regu_pwr);
        Y *= pow(1 - Y_lambda * Y_learning_rate * scale, regu_pwr);

        A_timebin *= pow(1 - A_timebin_lambda * A_timebin_learning_rate * scale, regu_pwr);
        B_timebin *= pow(1 - B_timebin_lambda * B_timebin_learning_rate * scale, regu_pwr);

        C_timebin *= pow(1 - C_timebin_lambda * C_timebin_learning_rate * scale, regu_pwr);

        for (int j = 0; j < A.n_cols; j++) {
            A.col(j) %= A_shrink; // Element wise multiplication
        }
        for (int j = 0; j < B.n_cols; j++) {
            B.col(j) %= B_shrink; // Element wise multiplication
        }        
    }

	void init(const record_array & train_data, int n_user, int n_movie) {

		// Calculate mu
		int cnt[6];
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
		date_origin_user = uvec(n_user, fill::zeros);
		date_origin_movie = uvec(n_movie, fill::zeros);

		uvec user_count(n_user, fill::zeros);
		uvec movie_count(n_movie, fill::zeros);
		for (int index = 0; index < train_data.size; index++) {
			int i = train_data[index].user - 1;
			int j = train_data[index].movie - 1;
			int d = train_data[index].date;
			user_count[i]++;
			movie_count[j]++;
			date_origin_user[i] += d;
			date_origin_movie[j] += d;
		}

		// Handle _count == 0
		for (int i = 0; i < user_count.n_elem; i++)
		{
			if (user_count[i] == 0) {
				date_origin_user[i] = 2243 / 2;
			} else {
				date_origin_user[i] /= user_count[i];
			}

		}
		for (int i = 0; i < movie_count.n_elem; i++)
		{
			if (movie_count[i] == 0) {
				date_origin_movie[i] = 2243 / 2;
			} else {
				date_origin_movie[i] /= movie_count[i];
			}
		}



		// Reshape the matrix based on n_user and n_movie
		U0.set_size(K, n_user);
		U1.set_size(K, n_user);
		V.set_size(K, n_movie);
		Y.set_size(K, n_movie);

		A_timebin.set_size(D_u, n_user);
		B_timebin.set_size(D_i, n_movie);
        C_timebin.set_size(D_u, n_user);


		function_table_generator ftg;
		vector<double> A_lambda_raw;
		vector<double> B_lambda_raw;

		// U table
		U1_function_table = ftg.abspwr_table(0.4);

		// A table

		A_function_table.insert_rows(A_function_table.n_rows, ftg.const_table());

		A_function_table.insert_rows(A_function_table.n_rows, ftg.abspwr_table(0.4));

		A_function_table.insert_rows(A_function_table.n_rows, ftg.abspwr_table(1));


		// B table
		
		B_function_table.insert_rows(B_function_table.n_rows, ftg.const_table());

		B_function_table.insert_rows(B_function_table.n_rows, ftg.abspwr_table(0.4));

		B_function_table.insert_rows(B_function_table.n_rows, ftg.abspwr_table(1));

		vector<double> w_list = { 2.0 * MAX_DATE / 28, 2.0 * MAX_DATE / 7, 2.0 * MAX_DATE / 90, 0.25, 1, 4};
		for (int i = 0; i < w_list.size(); i++) {
			double w = w_list[i];
			A_function_table.insert_rows(A_function_table.n_rows, ftg.sinw_table(i));

			A_function_table.insert_rows(A_function_table.n_rows, ftg.cosw_table(i));

			B_function_table.insert_rows(B_function_table.n_rows, ftg.sinw_table(i));

			B_function_table.insert_rows(B_function_table.n_rows, ftg.cosw_table(i));
		}

		A.resize(A_function_table.n_rows, n_user);
		B.resize(B_function_table.n_rows, n_movie);
		

		U0.fill(fill::randu);
		U1.fill(fill::zeros);
		V.fill(fill::randu);
		Y.fill(fill::zeros);

		A.fill(fill::zeros);
		B.fill(fill::zeros);
		A_timebin.fill(fill::zeros);
		B_timebin.fill(fill::zeros);
        C_timebin.fill(fill::zeros);

	}

	virtual void fit(const record_array & train_data, int n_iter, bool continue_fit = false) {
		try {
			int batch_size = 10000;
			int block_size = train_data.size / batch_size / 16;
			int n_user = 0, n_movie = 0;
			int *shuffle_idx;
			int *shuffle_idx_batch[N_THREADS];
			timer tmr;


			tmr.display_mode = 1;
			learning_rate_per_record = learning_rate;

			// Calculate n_user and n_movies
			for (int i = 0; i < train_data.size; i++) {
				if (train_data[i].user > n_user) {
					n_user = train_data[i].user;
				}
				if (train_data[i].movie > n_movie) {
					n_movie = train_data[i].movie;
				}
			}

			// Generate shuffle_idx
			cout << train_data.size << endl;

			shuffle_idx = new int[train_data.size / batch_size];
			for (int i = 0; i < train_data.size / batch_size; i++) {
				shuffle_idx[i] = i;
			}

			for (int i = 0; i < N_THREADS; i++) {

				shuffle_idx_batch[i] = new int[batch_size];
				for (int j = 0; j < batch_size; j++) {
					shuffle_idx_batch[i][j] = j;
				}
			}

			if (!continue_fit) {
				init(train_data, n_user, n_movie);
			}

            load_learning_rate(learning_rate_file);
            load_lambda(lambda_file);


			// Regenerate U_col
			for (int i = 0; i < N_THREADS; i++) {
				update_temp_var_thread[i].U_col.resize(K);
			}

			tmr.tic();
			// Regenerate the list of movies voted by one user
			vote_list.resize(n_user);
			for (int i = 0; i < n_user; i++) {
				vote_list[i] = vector<int>();
			}
			for (int i = 0; i < train_data.size; i++) {
				vote_list[train_data[i].user - 1].push_back(train_data[i].movie - 1);
			}
			tmr.toc();

			for (int i_iter = 0; i_iter < n_iter; i_iter++) {

				tmr.tic();
				cout << "Iter\t" << i_iter << '\t';


                rmse_sum = 0;
                rmse_count = 0;
				// Reshuffle first
				reshuffle(shuffle_idx, train_data.size / batch_size);

//#pragma omp parallel for num_threads(N_THREADS)
				for (int i = 0; i < train_data.size / batch_size; i++) {
					int index_base = shuffle_idx[i] * batch_size;

					reshuffle(shuffle_idx_batch[omp_get_thread_num()], batch_size);

					for (int j = 0; j < batch_size; j++) {
						int index = index_base + shuffle_idx_batch[omp_get_thread_num()][j];

						// shuffle_idx_batch[j] do harm to the result
						if (index < train_data.size) {
							const record& rcd = train_data[index];
							update(rcd, omp_get_thread_num());
						}
					}

					if ((i + 1) % block_size == 0) {
						cout << '.';                        
					}

                    if ((i + 1) % (train_data.size / batch_size / 16) == 0 && ((i + 1) % (train_data.size / batch_size) != 0))
                    {
                        regular();
                    }
				}
				if (ptr_test_data != NULL) {
					vector<float> result = this->predict_list(*ptr_test_data);
					cout << fixed;
					cout << setprecision(5);
					cout << '\t' << RMSE(*ptr_test_data, result);
                    cout << '\t' << rmse_sum / rmse_count;

                    char buf[256];
                    sprintf(buf, "probe_steps\\y%d.txt", i_iter);
#if _USE_Y
                    ofstream output_file(buf);

                    if (!output_file.is_open()) {
                        cerr << "Fail to open output file" << endl;
                        system("pause");
                        exit(-1);
                    }

                    for (int i = 0; i < result.size(); i++) {
                        output_file << result[i] << endl;
                    }
#endif
				}


				cout << '\t';
				tmr.toc();

                //cout << "----------------------------------------------------------" << endl;
                cout << setprecision(3);
				cout << "\t\t";
                cout << max(max(abs(U0))) << ' '
                    << max(max(abs(U1))) << ' '
                    << max(max(abs(V))) << ' '
                    << max(max(abs(Y))) << ' ' << endl;

                cout << "\t\t";
                cout << max(max(abs(A))) << ' '
                    << max(max(abs(B))) << ' '
                    << max(max(abs(A_timebin))) << ' '
                    << max(max(abs(B_timebin))) << ' '
                    << max(max(abs(C_timebin))) << endl;
                cout << endl;
                //cout << "----------------------------------------------------------" << endl;


				if (ptr_qual_data != NULL) {
					auto result = this->predict_list(*ptr_qual_data);

					cout << "Writting output file" << endl;

					char buf[256];
					sprintf(buf, "output_steps\\y%d.txt", i_iter);
#if _USE_Y
					ofstream output_file(buf);

					if (!output_file.is_open()) {
						cerr << "Fail to open output file" << endl;
						system("pause");
						exit(-1);
					}

					for (int i = 0; i < result.size(); i++) {
						output_file << result[i] << endl;
					}
#endif
				}

				if (i_iter != n_iter - 1) {
                    regular();
				}
                scale = scale * learning_rate_mul;
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


