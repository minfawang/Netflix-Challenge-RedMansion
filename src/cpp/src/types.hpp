#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <ctime>
#include <vector>
#include <omp.h>

#ifndef __DATA_TYPES
#define __DATA_TYPES

using namespace std;

class timer {
private:
	time_t last_time;
public:
	unsigned int display_mode;
public:
	timer() {
		display_mode = 0;
	}
	void tic() {
		last_time = clock();
	}
	void toc() {
		switch (display_mode)
		{
		case 0:
			cout << clock() - last_time << " ms" << endl;
			break;
		case 1:
			cout << (clock() - last_time) / 1000.0 << " s" << endl;
			break;
		}
		
	}
};

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
			delete [] data;
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

float RMSE(const record_array & test_data, const vector<float> & prediction) {
	double s = 0;
	for (int i = 0; i < test_data.size; i++) {
		s += (test_data[i].score - prediction[i]) * (test_data[i].score - prediction[i]);
	}
	return sqrt(s / test_data.size);
}

class estimator_base {
public:
	virtual bool save(const char * file_name) = 0;
	virtual bool load(const char * file_name) = 0;

	virtual void fit(const record_array & train_data, unsigned int n_iter = 1, bool countinue_fit = false) = 0;

	virtual float predict(const record & rcd) = 0;

	vector<float> predict_list(const record_array & rcd_array) {
		vector<float> result;
		result.resize(rcd_array.size);
#pragma omp parallel for num_threads(8)
		for (int i = 0; i < rcd_array.size; i++) {
			result[i] = predict(rcd_array[i]);
		}
		return result;
	}
};

class constant_estimator : public estimator_base {
public:
	float r;
	int cnt[6];

	void fit(const record_array & train_data) {
		long long sum = 0;

		for (int i = 0; i < 6; i++) {
			cnt[i] = 0;
		}

		for (int i = 0; i < train_data.size; i++) {
			cnt[int(train_data[i].score)]++;
		}

		sum = 0;
		for (int i = 0; i < 6; i++) {
			sum += cnt[i] * i;
		}
		r = 1.0 * sum / train_data.size;
	}

	float predict(const record & rcd) const {
		return r;
	}

};


#endif