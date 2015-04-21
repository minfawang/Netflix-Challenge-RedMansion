#include <iostream>
#include <fstream>
#include <sys/stat.h>

#ifndef __DATA_TYPES
#define __DATA_TYPES

using namespace std;

class record {

public:
	unsigned int user;
	unsigned int movie;
	unsigned int date;
	float score;

};

ostream & operator << (ostream &output, record& rcd) {
	output << rcd.user << " " << rcd.movie << " " << rcd.date << " " << rcd.score;
	return output;
}
istream & operator >> (istream &input, record& rcd) {
	input >> rcd.user >> rcd.movie >> rcd.date >> rcd.score;
	return input;
}

class record_array {
public:
	record * data;
	int size;

	record_array() {
		data = NULL;
		size = 0;
	}

	record_array(const char *file_name) {
		data = NULL;
		size = 0;
		load(file_name);
	}

	~record_array() {
		if (data != NULL) {
			delete data;
			data = NULL;
		}
	}

	const record & operator [] (unsigned int index) const{
		return data[index];
	}

	void load(const char *file_name) {
		if (data != NULL) {
			delete data;
			data = NULL;
		}

		struct stat results;
		// Get the size of the file
		if (stat(file_name, &results) != 0) {
			cout << "Stat Error" << endl;
			system("pause");
			exit(-1);
		}

		// Get the size of the array
		size = results.st_size / sizeof(record);

		// Allocate memory
		data = new record[size];

		ifstream data_from_file;
		data_from_file.open(file_name, ios::binary | ios::in);

		data_from_file.read((char *) data, size * sizeof(record));
	}
};

class estimator_base {
public:
	virtual void fit(const record_array & train_data) = 0;

	virtual float predict(const record & rcd) const = 0;

	vector<float> predict_list(const record_array & rcd_array) {
		vector<float> result;
		result.resize(rcd_array.size);
		for (int i = 0; i < rcd_array.size; i++) {
			result[i] = predict(rcd_array[i]);
		}
		return result;
	}
};

class constant_estimator : public estimator_base {
public:
	float r;
	void fit(const record_array & train_data) {
		for (int i = 0; i < train_data.size; i++) {
			r += train_data[i].score;
		}
		r /= train_data.size;	
	}

	float predict(const record & rcd) const {
		return r;
	}

};
#endif