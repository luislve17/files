/*
MD5
Algoritmo de encriptacion basada en Hashing
Documentacion utlizada:
	- https://www.ietf.org/rfc/rfc1321.txt
	- https://en.wikipedia.org/wiki/MD5
	
Desarrollador: Luis Vasquez Espinoza

Para compilar: g++ md5.cc -std=c++11 -o md5
Para ejecutar: ./md5
*/


#include <iostream>
#include <bitset>
#include <vector>
#include <sstream>
#include "md5.hpp"

using namespace std;
vector < bitset<32> > T = md5::init_CosTable();

/*
to_bytes_array()
input: String 'st'
output: Vector de bytes (cada celda es 8 bits) conteniendo cada caracter de 'st'
*/

vector< bitset<8> > md5::to_bytes_array(string st){
	vector< bitset<8> > bytes_vec(0);
	for(int i = 0; i < st.length(); i++){
		bitset<8> new_byte(st[i]);
		bytes_vec.push_back(new_byte);
	}
	return bytes_vec;
}

/*
append_padding_bits()
input: Vector de bytes 'bytes_vec'
output: Vector 'bytes_arr' con 10000...000 adjuntados (preparacion md5)

Si bien se debe adjuntar bits hasta que se cumpla que el numero de bits
resultantes es 448 mod 512, como estamos trabajando con vectores de bytes
realizaremos su equivalente, es decir: 56 mod 64
*/

void md5::append_padding_bits(vector < bitset<8> >* bytes_vec){
	int num_bytes = bytes_vec->size();

	bytes_vec->push_back(bitset<8>(string("10000000")));
	num_bytes++;
	
	while(num_bytes%64 != 56){
		bytes_vec->push_back(bitset<8>(string("00000000")));
		num_bytes++;
	}
}


/*
append_length()
input: Vector de bytes 'bytes_vec' previamente rellenados
output: Vector 'bytes_vec' con su longitud adjuntada

Segun la doc. se debe adjuntar su longitud empezando con los 32 bits
menos significativos primero, y luego los otros 32 bits mas significativos
*/


void md5::append_length(vector < bitset<8> >* bytes_vec, int original_size){
	string  original_size_bits = bitset<64>(original_size).to_string();
	
	// 32 bits menos y mas significativos de la expresion en 64 bits
	// de la long. original del vector
	vector< bitset<8> > least_significant(4); // vector de 4 celdas de 8 bits c/u
	vector< bitset<8> > most_significant(4); // vector de 4 celdas de 8 bits c/u
	
	
	for(int i = 0; i < 4; i++){ // para los 4 bytes en el vector 'most'
		most_significant[i] = bitset<8>(original_size_bits,8*i,8);
	}
	
	for(int i = 0; i < 4; i++){ // para los 4 bytes en el vector 'most'
		least_significant[i] = bitset<8>(original_size_bits,8*(i+4) ,8);
	}
	
	// Adjuntando los 32 bits de baja prioridad primero de atras hacia adelante
	for(int i = 3; i >= 0; i--){
		bytes_vec->push_back(least_significant[i]);
	}
	
	// Adjuntando los 32 bits de alta prioridad primero
	for(int i = 3; i >= 0; i--){
		bytes_vec->push_back(most_significant[i]);
	}	
}

/*
init_ABCDbuffer()
input: -
output: vector conteniendo los valores iniciales de A,B,C y D para MD5
*/
vector< bitset<32> > md5::init_ABCDbuffer(){
	bitset<32> A;
	bitset<32> B;
	bitset<32> C;
	bitset<32> D;
	
	A = 0x67452301;
	B = 0xefcdab89;
	C = 0x98badcfe;
	D = 0x10325476;
	
	vector < bitset<32> > buffer(0);

	buffer.push_back(A); buffer.push_back(B); buffer.push_back(C); buffer.push_back(D);
	
	return buffer;
}

/*
init_CosTable()
input: -
output: arreglo de valores de coseno (para MD5)

Posee numeros como 0xd76aa478, por lo que ocuparan 4 bytes cada uno (32 bits)
*/

vector < bitset<32> > md5::init_CosTable(){
	// Esta manera de inicializar un vector solo se soporta a partir de c++11
	vector < bitset<32> > tabla = {
							0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
							0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
							0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
							0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
							
							0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
							0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
							0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
							0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
							
							0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
							0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
							0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
							0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
							
							0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
							0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
							0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
							0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
							};
	return tabla;
}


/*
F(), G(), H() e I()
input: 3 valores de 32 bits c/u
output: valor de 32 bits resultante de operaciones logicas
*/

bitset<32> md5::F(bitset<32> X,bitset<32> Y, bitset<32> Z){
	bitset <32> result;
	result = X & Y | ~X & Z;
	return result;
}

bitset<32> md5::G(bitset<32> X,bitset<32> Y, bitset<32> Z){
	bitset <32> result;
	result = X & Z | Y & ~Z;
	return result;
}

bitset<32> md5::H(bitset<32> X,bitset<32> Y, bitset<32> Z){
	bitset <32> result;
	result = X ^ Y ^ Z;
	return result;
}

bitset<32> md5::I(bitset<32> X,bitset<32> Y, bitset<32> Z){
	bitset <32> result;
	result = Y ^ (X | ~Z);
	return result;
}

/*
CLS() : circular_left_shift
input: bitset de 32 bits
output: El bitset del input desfasado 't' unidades a la izquierda
*/

void md5::CLS(bitset<32> &x, int t){
	x = (x << t) | (x >> (32-t));
}

/*----------------------------- POR REVISAR -------------------------------*/

/*
FF(), GG(), HH(), II()
input: (ABCD, k, s, i), donde:
	ABCD: estado actual del buffer
	k:[0,15> Indice de una de las 16 partes del bloque del mensaje actual en operacion
	s:Cantidad de bits a rotar a la izquierda (circularmente)
	i:[0, 64> Indice del elemento i-esimo de la tabla de cosenos
	
Recordar que en cada paso, del mensaje original, se toma una sub_cadena de 512 bits,
la cual a la vez en cada sub_paso (los que se realizan en estas funciones FF,GG,...)
se operan de 32 en 32 bits (de la sub_cadena 512 = 16sub_sub_cadenas x 32 bits cada una)
*/

void md5::FF(bitset<32> &a, bitset<32> &b, bitset<32> &c, bitset<32> &d,
		int k, int s, int i, vector < bitset<32> > X){
	/* a = b + ((a + F(b,c,d) + X[k] + T[i]) <<< s) */
	if(0 <= k && k < 16 && 0 <= i && i < 64){
		a = a.to_ulong() + F(b, c, d).to_ulong() + X[k].to_ulong() + T[i].to_ulong();
		CLS(a, s);
		a = a.to_ulong() + b.to_ulong();
	} else {
		cout << "Error, k o i fuera de rango admisible" << endl;
	}
}

void md5::GG(bitset<32> &a, bitset<32> &b, bitset<32> &c, bitset<32> &d,
		int k, int s, int i, vector < bitset<32> > X){
	/* a = b + ((a + G(b,c,d) + X[k] + T[i]) <<< s) */
	if(0 <= k && k < 16 && 0 <= i && i < 64){
		a = a.to_ulong() + G(b, c, d).to_ulong() + X[k].to_ulong() + T[i].to_ulong();
		CLS(a, s);
		a = a.to_ulong() + b.to_ulong();
	} else {
		cout << "Error, k o i fuera de rango admisible" << endl;
	}
}

void md5::HH(bitset<32> &a, bitset<32> &b, bitset<32> &c, bitset<32> &d,
		int k, int s, int i, vector < bitset<32> > X){
	/* a = b + ((a + H(b,c,d) + X[k] + T[i]) <<< s) */
	if(0 <= k && k < 16 && 0 <= i && i < 64){
		a = a.to_ulong() + H(b, c, d).to_ulong() + X[k].to_ulong() + T[i].to_ulong();
		CLS(a, s);
		a = a.to_ulong() + b.to_ulong();
	} else {
		cout << "Error, k o i fuera de rango admisible" << endl;
	}
}
void md5::II(bitset<32> &a, bitset<32> &b, bitset<32> &c, bitset<32> &d,
		int k, int s, int i, vector < bitset<32> > X){
	/* a = b + ((a + I(b,c,d) + X[k] + T[i]) <<< s) */
	if(0 <= k && k < 16 && 0 <= i && i < 64){
		a = a.to_ulong() + I(b, c, d).to_ulong() + X[k].to_ulong() + T[i].to_ulong();
		CLS(a, s);
		a = a.to_ulong() + b.to_ulong();
	} else {
		cout << "Error, k o i fuera de rango admisible" << endl;
	}
}

/*
getNextChunk()
input: vector de bitset<8> y entero i
output: el i-esimo bitset de 64 bytes (512 bits) en el vector

La idea es que el vector msg entre con cada byte representando binariamente
cada caracter de un mensaje. Lo que se desea es que con 'i' accedamos al
i-esimo grupo de 512 bits dentro de 'msg', y lo retornemos 
*/
bitset<512>* md5::getNextChunk(vector< bitset<8> > msg, int i){
	if(64*(i+1) > msg.size()){ // Controlando que no nos salgamos del rango de 'msg'
		return NULL;
	}
	
	string aux = ""; // Strign auxiliar donde se empujaran los bytes
	for(int k = 0; k < 64; k++){
		aux += msg[64*i + k].to_string();
	}
	bitset<512>* result = new bitset<512>(aux);
	return result;
	
}


/*
breakChunk()
input: bitset de 512 bits
output: vector de bitsets 32 bits c/u, originarios del input
*/
vector< bitset<32> > md5::breakChunk(bitset<512> b){
	string aux = b.to_string();
	vector< bitset<32> > result(0);
		
	for(int i = 0; i < 16; i++){ // para los 16 grupos de 32 bits en 'b'
		result.push_back(bitset<32>(aux, 32*i,32));
	}
	
	return result;
}

/*
fixBigEndian()
input: bitset de 32 bit (digamos: A:00001111 B:10101100 C:11000101 D:01101110)
output: notacion little endian (del ejemplo: D:01101110 C:11000101 B:10101100 A:00001111)
*/
bitset<32> md5::fixBigEndian(bitset<32> current_32bit){
	string temp = current_32bit.to_string();

	string byte1 = temp.substr(0,8);
	string byte2 = temp.substr(8,8);
	string byte3 = temp.substr(16,8);
	string byte4 = temp.substr(24,8);

	return bitset<32>(byte4 + byte3 + byte2 + byte1);
}

/*
getHash()
input: vector de ABCD resultante del proceso md5
output: string representando el hash con un formato adecuado (imprime explicitamente las cifras ceros, por ejemplo)
*/

string md5::getHash(vector< bitset<32> > ABCD){
	string result = "";
	
	
	for(bitset<32> v: ABCD){ // Para A,B,C y D
		string aux;
		stringstream ss;
		ss << hex << uppercase << v.to_ulong();
		aux = ss.str();
		while(aux.size() < 8)
			aux.insert(0,"0"); // agregando ceros para registros pequenos
		result += aux;
	}
	return result;
}

string md5::get_md5(string st){
	vector< bitset<8> > test_bytes = to_bytes_array(st);
	append_padding_bits(&test_bytes);
	append_length(&test_bytes, 8*st.size());
	
	int i = 0;
	vector< bitset<32> > ABCD = init_ABCDbuffer();
	while(getNextChunk(test_bytes, i) != NULL){
		bitset<512> current_chunk = *getNextChunk(test_bytes, i);
		
		vector< bitset<32> > sub_chunks = breakChunk(current_chunk);
		vector< bitset<32> > doble_ABCD = ABCD; 
		
		/*
		== Nota del desarrollador == 
		Aqui tuve varios problemas debido a que el vector "suv_chunks", si bien sus elementos estaban
		correctamente instanciados, estaban en big_endian, y para que el algoritmo funcione debe estar
		en little_endian; es decir

		Digamos que "01000101 01010010 01010011 01001001" esta en el vector "sub_chunks", pero el algoritmo
		necesita que este como : "01001001 01010011 01010010 01000101", es decir, los octetos (bytes) en orden
		invertido, por lo que cree la funcion que se llama a continuacion.
		*/
		for(int i = 0; i < 16; i++){
			sub_chunks[i] = fixBigEndian(sub_chunks[i]);
		}

		/* Ronda 1
		[ABCD  0  7  1]  [DABC  1 12  2]  [CDAB  2 17  3]  [BCDA  3 22  4]
		[ABCD  4  7  5]  [DABC  5 12  6]  [CDAB  6 17  7]  [BCDA  7 22  8]
		[ABCD  8  7  9]  [DABC  9 12 10]  [CDAB 10 17 11]  [BCDA 11 22 12]
		[ABCD 12  7 13]  [DABC 13 12 14]  [CDAB 14 17 15]  [BCDA 15 22 16]
		*/
		
		FF(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 0, 7, 0, sub_chunks);
		FF(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 1, 12, 1, sub_chunks);
		FF(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 2, 17, 2, sub_chunks);
		FF(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 3, 22, 3, sub_chunks);
		
		FF(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 4, 7, 4, sub_chunks);
		FF(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 5, 12, 5, sub_chunks);
		FF(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 6, 17, 6, sub_chunks);
		FF(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 7, 22, 7, sub_chunks);
		
		FF(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 8, 7, 8, sub_chunks);
		FF(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 9, 12, 9, sub_chunks);
		FF(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 10, 17, 10, sub_chunks);
		FF(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 11, 22, 11, sub_chunks);
		
		FF(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 12, 7, 12, sub_chunks);
		FF(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 13, 12, 13, sub_chunks);
		FF(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 14, 17, 14, sub_chunks);
		FF(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 15, 22, 15, sub_chunks);
		
		/* Ronda 2
		[ABCD  1  5 17]  [DABC  6  9 18]  [CDAB 11 14 19]  [BCDA  0 20 20]
		[ABCD  5  5 21]  [DABC 10  9 22]  [CDAB 15 14 23]  [BCDA  4 20 24]
		[ABCD  9  5 25]  [DABC 14  9 26]  [CDAB  3 14 27]  [BCDA  8 20 28]
		[ABCD 13  5 29]  [DABC  2  9 30]  [CDAB  7 14 31]  [BCDA 12 20 32]
		*/
		GG(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 1, 5, 16, sub_chunks);
		GG(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 6, 9, 17, sub_chunks);
		GG(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 11, 14, 18, sub_chunks);
		GG(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 0, 20, 19, sub_chunks);
		
		GG(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 5, 5, 20, sub_chunks);
		GG(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 10, 9, 21, sub_chunks);
		GG(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 15, 14, 22, sub_chunks);
		GG(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 4, 20, 23, sub_chunks);
		
		GG(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 9, 5, 24, sub_chunks);
		GG(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 14, 9, 25, sub_chunks);
		GG(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 3, 14, 26, sub_chunks);
		GG(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 8, 20, 27, sub_chunks);
		
		GG(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 13, 5, 28, sub_chunks);
		GG(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 2, 9, 29, sub_chunks);
		GG(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 7, 14, 30, sub_chunks);
		GG(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 12, 20, 31, sub_chunks);
		
		/* Ronda 3
		[ABCD  5  4 33]  [DABC  8 11 34]  [CDAB 11 16 35]  [BCDA 14 23 36]
		[ABCD  1  4 37]  [DABC  4 11 38]  [CDAB  7 16 39]  [BCDA 10 23 40]
		[ABCD 13  4 41]  [DABC  0 11 42]  [CDAB  3 16 43]  [BCDA  6 23 44]
		[ABCD  9  4 45]  [DABC 12 11 46]  [CDAB 15 16 47]  [BCDA  2 23 48]
		*/
		
		HH(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 5, 4, 32, sub_chunks);
		HH(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 8, 11, 33, sub_chunks);
		HH(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 11, 16, 34, sub_chunks);
		HH(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 14, 23, 35, sub_chunks);
		
		HH(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 1, 4, 36, sub_chunks);
		HH(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 4, 11, 37, sub_chunks);
		HH(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 7, 16, 38, sub_chunks);
		HH(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 10, 23, 39, sub_chunks);
		
		HH(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 13, 4, 40, sub_chunks);
		HH(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 0, 11, 41, sub_chunks);
		HH(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 3, 16, 42, sub_chunks);
		HH(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 6, 23, 43, sub_chunks);
		
		HH(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 9, 4, 44, sub_chunks);
		HH(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 12, 11, 45, sub_chunks);
		HH(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 15, 16, 46, sub_chunks);
		HH(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 2, 23, 47, sub_chunks);
		/* Ronda 4
		[ABCD  0  6 49]  [DABC  7 10 50]  [CDAB 14 15 51]  [BCDA  5 21 52]
		[ABCD 12  6 53]  [DABC  3 10 54]  [CDAB 10 15 55]  [BCDA  1 21 56]
		[ABCD  8  6 57]  [DABC 15 10 58]  [CDAB  6 15 59]  [BCDA 13 21 60]
		[ABCD  4  6 61]  [DABC 11 10 62]  [CDAB  2 15 63]  [BCDA  9 21 64]		
		*/
		II(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 0, 6, 48, sub_chunks);
		II(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 7, 10, 49, sub_chunks);
		II(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 14, 15, 50, sub_chunks);
		II(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 5, 21, 51, sub_chunks);
		
		II(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 12, 6, 52, sub_chunks);
		II(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 3, 10, 53, sub_chunks);
		II(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 10, 15, 54, sub_chunks);
		II(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 1, 21, 55, sub_chunks);
		
		II(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 8, 6, 56, sub_chunks);
		II(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 15, 10, 57, sub_chunks);
		II(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 6, 15, 58, sub_chunks);
		II(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 13, 21, 59, sub_chunks);
		
		II(ABCD[0], ABCD[1], ABCD[2], ABCD[3], 4, 6, 60, sub_chunks);
		II(ABCD[3], ABCD[0], ABCD[1], ABCD[2], 11, 10, 61, sub_chunks);
		II(ABCD[2], ABCD[3], ABCD[0], ABCD[1], 2, 15, 62, sub_chunks);
		II(ABCD[1], ABCD[2], ABCD[3], ABCD[0], 9, 21, 63, sub_chunks);
		
		/*
		A = A + AA
		B = B + BB
		C = C + CC
		D = D + DD
		*/		
	
		ABCD[0] = ABCD[0].to_ulong() + doble_ABCD[0].to_ulong();
		ABCD[1] = ABCD[1].to_ulong() + doble_ABCD[1].to_ulong();
		ABCD[2] = ABCD[2].to_ulong() + doble_ABCD[2].to_ulong();
		ABCD[3] = ABCD[3].to_ulong() + doble_ABCD[3].to_ulong();
		
		i++;
	}

	ABCD[0] = fixBigEndian(ABCD[0]);
	ABCD[1] = fixBigEndian(ABCD[1]);
	ABCD[2] = fixBigEndian(ABCD[2]);
	ABCD[3] = fixBigEndian(ABCD[3]);
	
	return getHash(ABCD);
}