#include "hmac.hpp"
#include <bitset>
#include <vector>
#include <iostream>

using namespace std;

int main(){
	string msg = "what do ya want for nothing?";
	string key = "Jefe";
	cout << hmac::get_hmac(msg, key) << endl;
}