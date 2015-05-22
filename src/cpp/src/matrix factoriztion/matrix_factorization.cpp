#include <armadillo>
#include "../estimators/mf_estimators.hpp"
#include "../includes.hpp"

using namespace std;
using namespace arma;

int main(int argc, char * argv[]) {

    int n_iter = 40;

    char *output_name = "output\\output.txt";
    char *probe_output_name = "probe_output\\output.txt";

    char *learning_rate_file = "learning_rate.tmp";
    char *lambda_file = "lambda.tmp";

	if (argc == 6) {
        output_name = argv[1];
        probe_output_name = argv[2];
        learning_rate_file = argv[3];
        lambda_file = argv[4];
		n_iter = atoi(argv[5]);		        
	}

	record_array main, prob, qual;
	gamma_mf est;

    est.learning_rate_file = learning_rate_file;
    est.lambda_file = lambda_file;

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

	est.ptr_test_data = &prob;

#if !_USE_MINI_SET
    
	// est.ptr_qual_data = &qual;
#endif

	cout << "Start to fit" << endl;

	est.fit(main, n_iter);

	cout << "Start to predict" << endl;

	cout << "Prob set" << endl;

	vector<float> result = est.predict_list(prob);

	cout << "RMSE: " << RMSE(prob, result) << endl;

    float rmse = RMSE(prob, result);
    ofstream stat("stat.txt");
    if (isnan(rmse)){
        stat << 10.0 << endl;
    } else{
        stat << RMSE(prob, result) << endl;
    }
    

    ofstream probe_output_file(probe_output_name);
    for (int i = 0; i < result.size(); i++) {
        probe_output_file << result[i] << endl;
    }
#if !_USE_MINI_SET
	cout << "Qual set" << endl;

	result = est.predict_list(qual);

	cout << "Writting output file" << endl;

	ofstream output_file(output_name);

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


	//system("pause");
}