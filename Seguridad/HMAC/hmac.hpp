#ifndef HMAC_H
#define HMAC_H

#include <vector>
#include <bitset>

using namespace std;

class hmac{
	public:
	static vector< bitset<8> > init_ipad();
	static vector< bitset<8> > init_opad();
	static vector< bitset<8> > to_bytes_array(string st);
	static vector< bitset<8> > get_hex_values(string st);
	static void append_key_zeros(vector< bitset<8> > &key);
	static vector< bitset<8> > vector_xor(vector <bitset <8> > a, vector <bitset <8> > b);
	static string toString(vector< bitset<8> > v);
	static vector< bitset<8> > concat(vector< bitset<8> > a, vector< bitset<8> > b);
	static string get_hmac(string st, string key);
};

#endif