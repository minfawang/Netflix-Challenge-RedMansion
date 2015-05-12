#include <iostream>
#include "../types.hpp"


using namesapce std;


int main() {
	string train_file_name = "../../../data/mini_main.data";
	string test_file_name = "../../../data/mini_prob.data";


	record_array train_data;
	train_data.load(train_file_name.c_str());
	cout << "finish loading " << train_file_name << endl;

	record_array test_data;
	test_data.load(test_file_name.c_str());
	cout << "finish loading " << test_file_name << endl;

	record_array *ptr_train_data = &train_data;
	record_array *ptr_test_data = &test_data;

	unsigned int j = 0;
	unsigned int train_start = 0;
	unsigned int train_end = 0;
	unsigned int test_start = 0;
	unsigned int test_end = 0;
	unsigned int train_user = ptr_train_data->data[0].user;
	unsigned int test_user = ptr_test_data->data[0].user;

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


	return 0;

}