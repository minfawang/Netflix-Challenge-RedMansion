#include <iostream>
#include <stdlib.h>
#include <random>
#include "types.hpp"



#ifndef __MAKE_MINI_DATA
#define __MAKE_MINI_DATA

#define DIV 10
#define PROB_IN_MINI 20


using namespace std;






void split_mini (string prefix) {
	srand(1);
	string mini_name = "/Users/voiceup/Git/RedMansion-Netflix/src/data/" + prefix + "_data.data";
	string main_name = "/Users/voiceup/Git/RedMansion-Netflix/src/data/" + prefix + "_main.data";
	string prob_name = "/Users/voiceup/Git/RedMansion-Netflix/src/data/" + prefix + "_prob.data";
	ofstream mini_main, mini_prob;

	record_array recMini;
	recMini.load(mini_name.c_str());


	int prob_size = recMini.size / PROB_IN_MINI;
	int prob_idx[prob_size];
	short* main_prob_idx = new short[recMini.size];
	// int main_prob_idx[recMini.size];

	for (int i = 0; i < prob_size; ++i) {
		prob_idx[i] = i;
	}
	for (int i = 0; i < recMini.size; ++i) {
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

	mini_prob.close();
	mini_main.close();


	delete[] main_prob_idx;

}





void make_mini(string prefix) {
	char file_name[] = "/Users/voiceup/Git/RedMansion-Netflix/src/data/main_data.data";
	string fout_name = "/Users/voiceup/Git/RedMansion-Netflix/src/data/" + prefix + "_data.data";
	ofstream mini_main;

	record_array recMain;
	recMain.load(file_name);


	
	// size of main data is 98291670
	// cout << "The size: " << recMain.size << endl;


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
	string prefix = "mini";
	make_mini(prefix);
	split_mini(prefix);

	return 0;
}





#endif