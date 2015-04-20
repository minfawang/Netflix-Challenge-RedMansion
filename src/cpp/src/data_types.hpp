#include <iostream>
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
	output << rcd.user << " " << rcd.movie << " " << rcd.date << " " << rcd.score << endl;
	return output;
}
istream & operator >> (istream &input, record& rcd) {
	input >> rcd.user >> rcd.movie >> rcd.date >> rcd.score;
	return input;
}
#endif