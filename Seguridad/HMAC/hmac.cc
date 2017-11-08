/*
HMAC
Algoritmo de autenticacion de mensaje usando funciones hash de encriptacion (MD5)

Documentacion utlizada:
	- https://www.ietf.org/rfc/rfc2104.txt
	- https://es.wikipedia.org/wiki/HMAC#/media/File:SHAhmac.svg
	
Desarrollador: Luis Vasquez Espinoza

Para compilar: g++ md5.cc -std=c++11 -o md5
Para ejecutar: ./md5
*/

#include "hmac.hpp"
#include "../MD5/md5.hpp"
#include <bitset>
#include <vector>
#include <iostream>

using namespace std;

vector< bitset<8> > hmac::init_ipad(){
	vector< bitset<8> > ipad;
	for(int i = 0; i < 64; i++){
		ipad.push_back(0x36);
	}
	return ipad;
}

vector< bitset<8> > hmac::init_opad(){
	vector< bitset<8> > ipad;
	for(int i = 0; i < 64; i++){
		ipad.push_back(0x5C);
	}
	return ipad;
}

vector< bitset<8> > hmac::to_bytes_array(string st){
	vector< bitset<8> > bytes_vec(0);
	for(int i = 0; i < (int)st.length(); i++){
		bitset<8> new_byte(st[i]);
		bytes_vec.push_back(new_byte);
	}
	return bytes_vec;
}

vector< bitset<8> > hmac::get_hex_values(string st){
	vector< bitset<8> > bytes_vec(0);
	stringstream ss;
	int cont = 0;
	while(cont < (int)st.size()){
		unsigned int temp = stoul(st.substr(cont ,2), nullptr, 16);
		bytes_vec.push_back(bitset<8>(temp));
		cont += 2;
	}
	return bytes_vec;
}

void hmac::append_key_zeros(vector< bitset<8> > &key){
	while(key.size() < 64){
		key.push_back(0x00);
	}
}

vector< bitset<8> > hmac::vector_xor(vector <bitset <8> > a, vector <bitset <8> > b){
	vector< bitset<8> > result(64);
	for(int i = 0; i < 64; i++){
		result[i] = a[i]^b[i];
	}
	return result;
}

string hmac::toString(vector< bitset<8> > v){
	string result = "";
	for(auto i:v){
		result += (char)i.to_ulong();
	}
	return result;
}

vector< bitset<8> > hmac::concat(vector< bitset<8> > a, vector< bitset<8> > b){
	// a <- b : ab
	for(auto i: b){
		a.push_back(i);
	}
	return a;
}

template <typename T> void printVector(vector<T> v){
	for(auto i: v)
		cout << hex << (char)i.to_ulong();
	cout << endl;
}

string hmac::get_hmac(string st, string key){
	vector< bitset<8> > ipad, opad;
	ipad = hmac::init_ipad();
	opad = hmac::init_opad();

	auto st_bytes = hmac::to_bytes_array(st);
	auto key_bytes = hmac::to_bytes_array(key);
	
	if(key_bytes.size() > 64)
		key_bytes = hmac::to_bytes_array(md5::get_md5(hmac::toString(key_bytes)));

	hmac::append_key_zeros(key_bytes);
	auto ikey_pad = hmac::vector_xor(key_bytes, ipad);
	auto okey_pad = hmac::vector_xor(key_bytes, opad);
	
	string hashsum_1 = md5::get_md5(hmac::toString(hmac::concat(ikey_pad, st_bytes)));
	auto hashsum_1_bytes = hmac::get_hex_values(hashsum_1);
	string hashsum_2 = md5::get_md5(hmac::toString(hmac::concat(okey_pad, hashsum_1_bytes)));
	return hashsum_2;
}


