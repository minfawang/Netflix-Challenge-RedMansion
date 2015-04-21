#include <armadillo>
#include "..\estimators\mf_estimators.hpp"
#include "..\includes.hpp"

using namespace std;
using namespace arma;

int main(int argc, char * argv[]) {
	record_array main, prob, qual;
	basic_mf est;
	//constant_estimator est;

	time_t tic_time;
	time_t toc_time;

	tic_time = clock();
	main.load("main_data.data");
	prob.load("prob_data.data");
	qual.load("qual_data.data");

	cout << "Start to fit" << endl;
	est.fit(main);

	cout << "Start to predict" << endl;

	cout << "Prob set" << endl;

	vector<float> result = est.predict_list(prob);

	cout << MSE(prob, result) << endl;

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

	toc_time = clock();
	cout << result.size() << endl;
	cout << toc_time - tic_time << "ms" << endl;

	system("pause");
}