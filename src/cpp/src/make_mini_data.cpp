#include <iostream>
#include <stdlib.h>
#include <random>
#include "types.hpp"

using namespace std;
#define DIV 10
#define PROB_IN_MINI 20



void split_mini () {
	srand(1);
	char main_name[] = "/Users/voiceup/Git/RedMansion-Netflix/src/data/mini_main.data";
	char prob_name[] = "/Users/voiceup/Git/RedMansion-Netflix/src/data/mini_prob.data";
	char mini_name[] = "/Users/voiceup/Git/RedMansion-Netflix/src/data/mini_data.data";
	ofstream mini_main, mini_prob;

	record_array recMini;
	recMini.load(mini_name);

	int maxU = 45829;
	int maxM = 1777;
	int prob_size = recMini.size / PROB_IN_MINI;
	int prob_idx[prob_size];
	int main_prob_idx[recMini.size];
	for (int i = 0; i < prob_size; i++) {
		prob_idx[i] = i;
	}
	for (int i = 0; i < recMini.size; i++) {
		main_prob_idx[i] = 0;
	}

	for (int i = prob_size; i < recMini.size; i++) {
		int rand_num = rand() % i;
		if (rand_num < prob_size) {
			prob_idx[rand() % prob_size] = i;
		}
	}

	for (int i = 0; i < prob_size; i++) {
		main_prob_idx[prob_idx[i]] = 1;
	}


	mini_main.open(main_name, ios::binary | ios::out);
	mini_prob.open(prob_name, ios::binary | ios::out);
	for (int i = 0; i < recMini.size; i++) {
		record r = recMini.data[i];
		if (main_prob_idx[i])
			mini_prob.write((char*)&r, sizeof(r));
		else
			mini_main.write((char*)&r, sizeof(r));
	}

}





void make_mini() {
	char file_name[] = "/Users/voiceup/Git/RedMansion-Netflix/src/data/main_data.data";
	char fout_name[] = "/Users/voiceup/Git/RedMansion-Netflix/src/data/mini_data.data";
	ofstream mini_main;

	record_array recMain;
	recMain.load(file_name);


	
	// size of main data is 98291670
	// cout << "The size: " << recMain.size << endl;		

	// int maxU = 0;
	// int maxM = 0;
	// for (int i = 0; i < recMain.size; i++) {
	// 	record r = recMain.data[i];
	// 	if (r.user > maxU)
	// 		maxU = r.user;
	// 	if (r.movie > maxM)
	// 		maxM = r.movie;
	// }

	// cout << "Max user: " << maxU << endl;
	// cout << "Max movie: " << maxM << endl;

	int maxU = 458293 / DIV;
	int maxM = 17770 / DIV;

	mini_main.open(fout_name, ios::binary | ios::out);
	for (int i = 0; i < recMain.size; i++) {
		record r = recMain.data[i];
		if ((r.user < maxU) && (r.movie < maxM)) {
			mini_main.write((char *)&r, sizeof(r));
		}
	}

}



int main () {
	// make_mini();
	// split_mini();

	return 0;
}