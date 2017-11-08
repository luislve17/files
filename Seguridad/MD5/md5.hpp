#ifndef MD5_H
#define MD5_H

#include <iostream>
#include <bitset>
#include <vector>
#include <sstream>

using namespace std;

class md5{
	public:
	
	static vector< bitset<8> > to_bytes_array(string st);
	static void append_padding_bits(vector< bitset<8> >* bytes_vec);
	static void append_length(vector < bitset<8> >* bytes_vec, int original_size);
	
	static vector< bitset<32> > init_ABCDbuffer();
	static vector < bitset<32> > init_CosTable();
	
	static bitset<32> F(bitset<32> X,bitset<32> Y, bitset<32> Z);
	static bitset<32> G(bitset<32> X,bitset<32> Y, bitset<32> Z);
	static bitset<32> H(bitset<32> X,bitset<32> Y, bitset<32> Z);
	static bitset<32> I(bitset<32> X,bitset<32> Y, bitset<32> Z);
	
	static void CLS(bitset<32> &bits, int t);
	static bitset<32> fixBigEndian(bitset<32> current_32bit);
	static void FF(bitset<32> &a, bitset<32> &b, bitset<32> &c, bitset<32> &d,
			int k, int s, int i, vector < bitset<32> > X);
	static void GG(bitset<32> &a, bitset<32> &b, bitset<32> &c, bitset<32> &d,
			int k, int s, int i, vector < bitset<32> > X);
	static void HH(bitset<32> &a, bitset<32> &b, bitset<32> &c, bitset<32> &d,
			int k, int s, int i, vector < bitset<32> > X);
	static 	void II(bitset<32> &a, bitset<32> &b, bitset<32> &c, bitset<32> &d,
			int k, int s, int i, vector < bitset<32> > X);
			
	static 	bitset<512>* getNextChunk(vector< bitset<8> > msg, int i);
	static 	vector< bitset<32> > breakChunk(bitset<512> b);
	static 	string getHash(vector< bitset<32> > ABCD);

	static string get_md5(string st);
};

#endif