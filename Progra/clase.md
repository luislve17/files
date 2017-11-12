# Tema : Manipulación de archivos

> A resolver:

* ¿Cómo se representa un archivo en C?
* ¿Qué importancia tiene saber manipular archivos?

## 1. Introducción
Existen muchas maneras de almacenar información en un programa en C. Hasta ahora solo se han revisado estructuras de datos como los __arreglos__ y las __estructuras__ pero ninguna de estas nos proporciona información útil acabada la ejecución del programa. Es por esta razón que la manipulación de archivos sugiere una herramienta poderosa y flexible.

La idea de su implementación busca salvar algún dato útil resultado de la ejecución de una aplicación que no podría ser manipulado de otra manera.

## 2. Sintaxis básica

```c
#include <stdio.h>

int main(){
	FILE* fptr; // puntero a un archivo
	int my_num = 10;
	// fopen(nombre_archivo, tipo_apertura)
	fptr = fopen("archivo_creado.txt", "w");

	// comprobamos si el archivo se creó correctamente
	if(fptr == NULL){
		printf("Error. No se pudo obtener archivo.\n");
		return -1;	
	}

	// Imprimimos algo en el archivo
	fprintf(fptr, "%d", my_num);
	// Cerramos el archivo
	fclose(fptr);

	return 0;
}
```

## 3. Tipos de modos de apertura
<center>
<font size = "3">

|MODO  | Descripción | Manejo de error |
|:----:|:-----------:|:---------------:|
|r|Lectura|fopen() retorna NULL si el file no existe|
|rb|Lectura binaria|fopen() retorna NULL si el file no existe|
|w|Escritura|Si el file existe, su contenido se sobreescribe. Si no, se crea uno nuevo|
|wb|Escritura binaria|Si el file existe, su contenido se sobreescribe. Si no, se crea uno nuevo|
|a|Anexión|Si el file no existe, se crea uno nuevo|
|ab|Anexión binaria|Si el file no existe, se crea uno nuevo|
|r+|Escritura + Lectura|fopen() retorna NULL si el file no existe|
|rb+|Escritura + Lectura binaria|fopen() retorna NULL si el file no existe|
|w+|Escritura + Lectura|Si el file existe, su contenido se sobreescribe. Si no, se crea uno nuevo|
|wb+|Escritura + Lectura binaria|Si el file existe, su contenido se sobreescribe. Si no, se crea uno nuevo|
|a+|Anexión + Lectura|Si el file no existe, se crea uno nuevo|
|ab+|Anexión + Lectura binaria|Si el file no existe, se crea uno nuevo|

</font>
</center>

## 4. Implementaciones útiles
### 4.1 Seek y fgetc
El siguiente ejemplo implementa una manera de mover el puntero _FILE *_ de un archivo para posicionarlo en un lugar arbitrario de nuestra preferencia; además, veremos como leer el caracter al que apunta este puntero en un instante especifico

```c
/*
Funciones:
	... fscanf(FILE* file_ptr, string formato, ...)
	int fseek(FILE * file_ptr, long int offset, int pivote) // 0: exito
	... feof(FILE * file_ptr)
*/

#include <stdio.h>

int main(){
	FILE * fptr = fopen("por_leer.txt", "r"); // Abrimos el archivo en modo lectura

	for(int i = 1; i <= 4; i++){ // Leeremos los primeros 4 caracteres del file
		char ch; // Para almacenar el i-esimo caracter
		fscanf(fptr, "%c", &ch); // Escaneamos del archivo
		printf("%c|", ch); // Imprimimos en terminal
	}

	char ch_2;
	fscanf(fptr, "%c", &ch_2);
	printf("\n%c\n", ch_2);

	/*=========== Pausa de explicacion ===========*/ getchar();

	// Moviendo el puntero de file
	fseek(fptr, 3, SEEK_SET);
	char ch_3;
	fscanf(fptr, "%c", &ch_3);
	printf("Luego del seek(), leemos: %c\n", ch_3);

	fseek(fptr, -4, SEEK_END);
	char ch_4;
	fscanf(fptr, "%c", &ch_4);
	printf("Luego del segundo seek(), leemos: %c\n", ch_4);

	/*=========== Pausa de explicacion ===========*/ getchar();
	fseek(fptr, 0, SEEK_SET);
	while(!feof(fptr)){
		char ch_iterando;
		fscanf(fptr, "%c", &ch_iterando);
		printf("%c,", ch_iterando);
	}

	printf("\n");
	
	fseek(fptr, 0, SEEK_SET);
	char ch_iterando;
	fscanf(fptr, "%c", &ch_iterando);
	while(!feof(fptr)){
		printf("%c,", ch_iterando);
		fscanf(fptr, "%c", &ch_iterando);
	}

	printf("\n");
	return 0;
}
```

### 4.2 fscanf con formato
Así como cuando se utilizaba _scanf_ para definir una entrada __formateada__, se puede formatear una entrada obtenida de un archivo para manipular el texto como valores operables. En el ejemplo tomamos la primera linea del archivo como formato preparado para leer los tres números presentes.
```c
/*
Funciones:
	... fscanf(FILE* file_ptr, string formato, ...)
	int fseek(FILE * file_ptr, long int offset, int pivote) // 0: exito
	... feof(FILE * file_ptr)
*/

#include <stdio.h>

int main(){
	FILE * fptr = fopen("por_leer.txt", "r"); // Abrimos el archivo en modo lectura
	
	if(fptr == NULL){ // Manejo de error
		printf("Error. Archivo no encontrado\n");
		return -1;
	}
	
	char cadena[100];
	while(!feof(fptr)){
		fscanf(fptr, "%s", cadena);
		printf("%s|", cadena);
	}
	printf("\n");

	/*=========== Pausa de explicacion ===========*/ getchar();
	fseek(fptr, 0, SEEK_SET);
	int x1, x2, x3, status;
	fscanf(fptr, "Numeros a sumar: %d, %d y %d", &x1, &x2, &x3);
	printf("%d+%d+%d=%d\n", x1, x2, x3, x1+x2+x3);	
	return 0;
}
```

## 5. Ejercicios
1. Manejo de modos de manipulación de archivos
* 1.1 Del archivo existente denominado "file_existente.txt" manipule el archivo de tal manera que sin modificar nada agregue los números del 0 al 50 (uno por cada línea)

2. Manipulación general de archivos
* 2.1 Usando un puntero __FILE * input__ abra un archivo en modo _escritura_ y almacene en dicho archivo datos numéricos (double) ingresados por el usuario (un dato numerico por linea), mientras que este no ingrese un numero negativo.

* 2.2 Usando un puntero __FILE * output__ abra el archivo generado anteriormente en modo _lectura_ y acumule cada uno de los numeros obtenidos de dicho archivo a una variable numerica acumuladora

* 2.3 Imprima la suma acumulada total

___
Nota:
Dado que el ultimo elemento es negativo, puede que al leer los datos acabe acumulando dicho negativo a la suma total, lo cual no se desea.
___

3. Manipulacion de archivos, estructuras y paso por valor y referencia
* 3.1 Defina una estructura __'empleado'__ que presente 4 campos: nombre del empleado(char[100]), su area de trabajo(char [100]), su edad(int) y salario(double)

* 3.2 Implemente el prototipo: __void ingresarCadena(char *cad)__, donde se ingresa __por referencia__ una cadena de caracteres y mediante el uso de la función __getchar()__ el usuario es capaz de ingresar una cadena de longitud no mayor a 100. (no olvidar el caracter nulo al final)

* 3.3 Implemente el prototipo: __void printCadena(char cad[100])__, donde se ingresa __por valor__ una cadena de caracteres y se imprime uno a uno sus elementos

* 3.4 Implemente el prototipo __void generarFicha(struct empleado emp)__, donde se ingresa como argumento un empleado y se genera un archivo llamado __'ficha.txt'__ conteniendo los datos del empleado (un dato por línea)

## 5. Soluciones
### 1.
```c
#include <stdio.h>

int main(){
	// ---------------- Parte 1 ----------------
	FILE* input = fopen("file_existente.txt", "a");
	
	for(int i = 0; i < 50; i++){
		fprintf(input, "%d\n", i);
	}

	fclose(input);
}
```


### 2.
```c
#include <stdio.h>

int main(){
	// ---------------- Parte 1 ----------------
	FILE* input = fopen("datos.txt", "w");
	
	int cont = 1;
	double dato_actual;
	do{
		printf("Ingrese dato #%d (o valor negativo para salir): ", cont++);
		scanf("%lf", &dato_actual);
		fprintf(input, "%.2lf\n", dato_actual);
	}while(dato_actual >= 0);
	fclose(input);

	printf("Archivo generado!\n");
	getchar();
	// ---------------- Parte 2 ----------------
	FILE* output = fopen("datos.txt", "r");

	if(output == NULL){
		printf("Error. Archivo no encontrado\n");
		return -1;
	}

	double dato_recibido;
	double acc = 0;
	while(--cont > 1){
		fscanf(output, "%lf\n", &dato_recibido);
		acc += dato_recibido;
	}
	printf("total = %.2lf\n", acc);
	return 0;
}
```
### 3.
```c
#include <stdio.h>

struct empleado{
	char nombre[100];
	char area_trabajo[100];
	int edad;
	double salario;
};

void ingresarCadena(char *cad);
void printCadena(char cad[100]);
void generarFicha(struct empleado emp);

int main(){
	struct empleado emp_1;
	printf("Nombre: ");
	ingresarCadena(emp_1.nombre);
	printf("\nArea: ");
	ingresarCadena(emp_1.area_trabajo);
	printf("\nEdad: ");
	scanf("%d", &emp_1.edad);
	printf("\nSalario: ");	
	scanf("%lf", &emp_1.salario);

	generarFicha(emp_1);

	return 0;
}

void ingresarCadena(char* cad){
	int cont = 0;
	char input = getchar();
	while(input != '\n'){
		cad[cont] = input;
		input = getchar();
		cont++;
	}
	cad[cont] = '\0';
}

void printCadena(char cad[100]){
	char * pointer = cad;
	while(*pointer != '\0')
		printf("%c", *pointer++);
}

void generarFicha(struct empleado emp){
	FILE* ptr_ficha;
	ptr_ficha = fopen("ficha.txt", "w");
	if(ptr_ficha == NULL){
		printf("Error. No se pudo generar ficha");
		return;
	}
	fprintf(ptr_ficha, "%s\n", emp.nombre);
	fprintf(ptr_ficha, "%s\n", emp.area_trabajo);
	fprintf(ptr_ficha, "%d\n", emp.edad);
	fprintf(ptr_ficha, "%.2lf", emp.salario);
	fclose(ptr_ficha);
}
```