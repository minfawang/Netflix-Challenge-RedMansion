#include <armadillo>
#include "../estimators/mf_estimators.hpp"
#include "../includes.hpp"

using namespace std;
using namespace arma;

int main(int argc, char * argv[]) {

	unsigned int n_iter = 30;
	float learning_rate = 0.008;
	float lambda_factor = 1000000;
	float U0_lambda = 0.00008;
	float U1_lambda = 0.00008;
	float V_lambda = 0.00008;
	float Y_lambda = 0.00008;
	float lambda = 0.0001;

	if (argc == 8) {
		n_iter = atoi(argv[1]);
		learning_rate = atof(argv[2]);
		lambda_factor = atof(argv[3]);
		U0_lambda = atof(argv[4]);
		U1_lambda = atof(argv[5]);
		V_lambda = atof(argv[6]);
		lambda = atof(argv[7]);
	}

	record_array main, prob, qual;
	gamma_mf est;

	//constant_estimator est;

	time_t tic_time;
	time_t toc_time;

	tic_time = clock();

#define _USE_MINI_SET 1
#define _TEST_SAVE_AND_LOAD 0


#if _USE_MINI_SET
	main.load("mini_main.data");
	prob.load("mini_prob.data");
#else
	main.load("main_data.data");
	prob.load("prob_data.data");
	qual.load("qual_data.data");
#endif

	est.learning_rate = learning_rate;
	est.lambda_factor = lambda_factor;

	est.U0_lambda = U0_lambda;
	est.U1_lambda = U1_lambda;
	est.V_lambda = V_lambda;
	est.Y_lambda = Y_lambda;
	est.lambda = lambda;

	est.ptr_test_data = &prob;

	cout << "Start to fit" << endl;

	est.fit(main, n_iter);

	cout << "Start to predict" << endl;

	cout << "Prob set" << endl;

	vector<float> result = est.predict_list(prob);

	cout << "RMSE: " << RMSE(prob, result) << endl;

#if !_USE_MINI_SET
	cout << "Qual set" << endl;

	result = est.predict_list(qual);

	cout << "Writting output file" << endl;

	ofstream output_file("output\\output.txt");

	if (!output_file.is_open()) {
		cerr << "Fail to open output file" << endl;
		system("pause");
		exit(-1);
	}

	for (int i = 0; i < result.size(); i++) {
		output_file << result[i] << endl;
	}
#endif

#if _TEST_SAVE_AND_LOAD
	cout << "Saving Model to test_model_files/test_model.data" << endl;
	est.save("test_model_files/test_model.data");

	alpha_mf est2;
	cout << "Loading Model from test_model_files/test_model.data" << endl;
	est2.load("test_model_files/test_model.data");

	result = est2.predict_list(prob); 
	cout << "RMSE: " << RMSE(prob, result) << endl;

	est2.fit(main, 5, true);
	result = est2.predict_list(prob);
	cout << "RMSE: " << RMSE(prob, result) << endl;

#endif

	toc_time = clock();
	cout << result.size() << endl;
	cout << toc_time - tic_time << "ms" << endl;

	system("pause");
}