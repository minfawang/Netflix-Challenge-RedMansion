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
	record_array mini_main, mini_prob;
	// string main_file_name = "../../data/main_data.data";
	// string prob_file_name = "../../data/prob_data.data";
	string main_file_name = "../../data/mini_main.data";
	string prob_file_name = "../../data/mini_prob.data";
	mini_main.load(main_file_name.c_str());
	mini_prob.load(prob_file_name.c_str());


	int count_user_no_rating = 0;

	int im = 0;
	int ip = 0;

	// int main_user = mini_main.data[0].user;
	int prob_user = mini_prob.data[0].user;

	while (ip < mini_prob.size) {
		record prob_r = mini_prob.data[ip];
		// when there is a new user in prob
		if (prob_r.user != prob_user) {
			while (im < mini_main.size) {
				record main_r = mini_main.data[im];
				if (main_r.user > prob_user) {
					count_user_no_rating++;
					break;
				} else if (main_r.user == prob_user) {
					break;
				}
				im++;
			}
			

			// update cur prob user
			prob_user = prob_r.user;
		}

		ip ++;
	}


	cout << "number no rating = " << count_user_no_rating << endl;


	return 0;
}