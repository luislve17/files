#include "md5.hpp"

// ======================== MAIN =====================================
int main(){
	// Vectores de prueba de wikipedia (https://en.wikipedia.org/wiki/MD5)
	string test_1 = "The quick brown fox jumps over the lazy dog";
	//string test_2 = "The quick brown fox jumps over the lazy dog.";
	//string test_3 = "";
	
	// Vectores de prueba vistos en clase
	//string test_4 = "UNIVERSIDAD NACIONAL DE INGENIERIA0";
	//string test_5 = "UNIVERSIDAD NACIONAL DE INGENIERIA81160";
	//string test_6 = "UNIVERSIDAD NACIONAL DE INGENIERIA12604094";

	cout << "Hash de \"" << test_1 << "\": " << md5::get_md5(test_1) << endl;
	//cout << "Hash de \"" << test_2 << "\": " << md5::digest(test_2) << endl;
	//cout << "Hash de \"" << test_3 << "\": " << md5::digest(test_3) << endl;
	//cout << "Hash de \"" << test_4 << "\": " << md5::digest(test_4) << endl;
	//cout << "Hash de \"" << test_5 << "\": " << md5::digest(test_5) << endl;
	//cout << "Hash de \"" << test_6 << "\": " << md5::digest(test_6) << endl;
}
// ======================== MAIN =====================================
