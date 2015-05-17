#include <iostream>
#include <unordered_map>
#include <cmath>
#include "../types.hpp"


using namespace std;



void write_processed_data_to_file( unordered_map<unsigned int, int*> user_map, ofstream &out_file) {

	for (int i = 0; i < user_map.size(); i++) {
		// TODO
	}
}


// user: [start, end]
unordered_map<unsigned int, int*> make_pre_map(const char* file_name) {
	unordered_map<unsigned int, int*> record_map;

	record_array record_data(file_name);
	cout << "finish loading " << file_name << endl;
	unsigned int cur_user = record_data.data[0].user;
	int cur_start = 0;
	int cur_end = 1;
	int* user_ids;
	for (int i = 0; i < record_data.size; i++) {
		record this_data = record_data.data[i];
		if (this_data.user != cur_user) {
			cur_end = i;
			

			user_ids = new int[2];
			user_ids[0] = cur_start;
			user_ids[1] = cur_end;
			record_map[cur_user] = user_ids;
			
			cur_user = this_data.user;
			cur_start = i;
		}
	}
	user_ids = new int[2];
	user_ids[0] = cur_start;
	user_ids[1] = record_data.size;
	record_map[cur_user] = user_ids;

	cout << "number of users = " << record_map.size() << endl;

	return record_map;
}


int main() {
	string train_file_name = "../../../data/main_data.data";
	string test_file_name = "../../../data/prob_data.data";
	string qual_file_name = "../../../data/qual_data.data";



	string out_file_name = "../../../data/data.pre";
	ofstream out_file;
	out_file.open(out_file_name);



	/* TEST CDOE */
	// string test_str = "1 3 5 7\n2 10 3249";
	int test_str[3];
	test_str[0] = 0;
	test_str[1] = 1;
	test_str[2] = 2;

	string test_out_file_name = "test_vec.txt";
	string line;
	vector <string> lines;
	ofstream test_out_file(test_out_file_name.c_str(), ios::binary | ios::out);

	test_out_file << test_str;
	test_out_file.close();

	struct stat results;
	if (stat(file_name, &results) != 0) {
		cout << "Stat Error" << endl;
		exit(-1);
	}


	ifstream test_in_file(test_out_file_name.c_str(), ios::binary | ios::in);
	while (getline(test_in_file, line)) {
		lines.push_back(line);
	}

	// for (vector<string>::iterator it = lines.begin(); it != lines.end(); ++it) {
	// 	cout << *it << endl;
	// }

	// test_in_file.close();




	unordered_map<unsigned int, int*> train_map = make_pre_map(train_file_name.c_str());
	unordered_map<unsigned int, int*> test_map = make_pre_map(test_file_name.c_str());
	unordered_map<unsigned int, int*> qual_map = make_pre_map(qual_file_name.c_str());


	// for (auto user_ids = train_map.begin(); user_ids != train_map.end(); ++user_ids) {
	for (unsigned int user = 1; user < train_map.size() + 1; user++) {

		int* ids = train_map[user];

		out_file << user << " " << ids[0] << " " << ids[1] << endl;
		
		unordered_map<unsigned int, int*>::const_iterator test_ids = test_map.find(user);
		if (test_ids != test_map.end()) {
			ids = test_ids -> second;
			out_file << user << " " << ids[0] << " " << ids[1] << endl;
		}

		unordered_map<unsigned int, int*>::const_iterator qual_ids = qual_map.find(user);
		if (qual_ids != qual_map.end()) {
			ids = qual_ids -> second;
			out_file << user << " " << ids[0] << " " << ids[1] << endl;
		}

	}

	// // write_processed_data_to_file(train_vec, out_file);
	// // write_processed_data_to_file(test_vec, out_file);
	// // write_processed_data_to_file(qual_vec, out_file);

	// out_file.close();

	return 0;
}