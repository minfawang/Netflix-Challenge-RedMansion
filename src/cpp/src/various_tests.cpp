#include <iostream>
#include <cmath>
#include "types.hpp"

using namespace std;

int main() {
	/* print to see what data is in mini_main */
	// record_array mini_main;
	// string file_name = "/Users/voiceup/Git/RedMansion-Netflix/src/data/mini_main.data";
	// mini_main.load(file_name.c_str());
	// for (int i = 0; i < 100; i++) {
	// 	record r = mini_main.data[i];
	// 	cout << r.user << endl;
	// }


	// /* export mini_prob to test_mini_prob.txt*/
	// record_array mini_prob;
	// string file_name = "../../data/mini_prob.data";
	// mini_prob.load(file_name.c_str());

	// ofstream out_file("test_mini_prob.txt");
	
	// for (int i = 0; i < mini_prob.size; i++) {
	// 	record r = mini_prob.data[i];
	// 	out_file << r.score << endl;
	// }



	/* test the percentage of user didn't rate any movie */
	record_array main_array, prob_array;
	// string main_file_name = "../../data/main_data.data";
	// string prob_file_name = "../../data/prob_data.data";
	string main_file_name = "../../data/mini_main.data";
	string prob_file_name = "../../data/mini_prob.data";
	main_array.load(main_file_name.c_str());
	prob_array.load(prob_file_name.c_str());

	vector<float> results;
	results.resize(prob_array.size);

	int count_user_no_rating = 0;

	int im = 0;
	int ip = 0;

	int test_start = 0;
	int test_end = 0;
	int train_start = 0;
	int train_end = 0;

	// int main_user = main_array.data[0].user;
	int prob_user = prob_array.data[0].user;

	while (ip < prob_array.size) {
		record prob_r = prob_array.data[ip];
		// when there is a new user in prob
		if (prob_r.user != prob_user) {

			test_end = ip;

			// record main_r;
			// main_r.user = 0;
			while (im < main_array.size) {
				record main_r = main_array.data[im];
				if (main_r.user > prob_user) {
					break;
				} else if (main_r.user < prob_user) {
					train_start++;
				}
				im++;
			}

			// found a user match
			if (main_array.data[im-1].user == prob_user) {
				train_end = im;
				for (int u = test_start; u < test_end; u++) {
					results[u] = 0;
				}

			} else {
				count_user_no_rating++;
				for (int u = test_start; u < test_end; u++) {
					results[u] = 3.6;
				}
			}
			
			train_start = im;

			// update cur prob user
			prob_user = prob_r.user;
			test_start = ip;
		}

		ip ++;
	}


	cout << "number no rating = " << count_user_no_rating << endl;


	return 0;
}