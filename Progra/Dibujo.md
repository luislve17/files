# Problema de impresión/dibujo en terminal (implementación)
## Ejemplo de desarrollo
**Ejercicio 2.4:** Desarrolle un programa que pida ingresar un entero n > 1 y dibuje un triángulo de altura 2n − 1 y ancho n. Por ejemplo, para n = 2 se tendría:


```
terminal~

*
* *
*
```

### 1. Idea:
Para todo problema de este tipo debemos de revisar primero el dibujo que piden. Del ejercicio sabemos que el resultado se escribirá en el terminal en dos dimensiones, por lo que para implementarlo necesitaremos _dos estructuras iterativas anidadas una en otra_; y será bajo una condición que escribiremos los asteriscos oportunamente:

```
para(y en [-1;1]){
	para(x en [-2;2]){
		si(y < |x|){
			dibujar asterisco en (x, y)
		} sino {
			dibujar ESPACE en (x, y)
		}
	}
}
```

Este pseudocodigo resultaría en la graficación de la región debajo de la curva _valor absoluto_ (aproximadamente)

```
 1 |*|_|_|_|*|
 0 |*|*|_|*|*|
-1 |*|*|*|*|*|
   -2 -1 0 1 2
```

Entonces, nuestra tarea al resolver un problema de este tipo se resume a tres aspectos:
* Definir en que rango de 'y' (vertical) deseamos dibujar
* Definir en que rango de 'x' (horizontal) deseamos dibujar
* Controlar la estructura if dentro del loop interno para saber la regla que domina el criterio de graficación

### 2. Implementación:
Regresando al ejercicio, notamos que es el usuario el que tácitamente define el rango x e y. Si el usuario ingresa 'n' la altura del triangulo (verticalmente) sería de '2n - 1', y horizontalmente ocuparía 'n' unidades. Esta primera idea nos debería ayudar a concluir que los rangos de los for's podrían definirse en **y = [-(n - 1);n - 1]** | **x = [1;n]**, para cualquier n (si aún tiene dificultades en saber por qué es conveniente dar estos valores a los rangos, podría intentar remplazar n por un número cualquiera y revisar el sentido otra vez). Ahora solo quedaría especificar la condición de graficación.

```c
#include <stdio.h>

int main(){
	// Asumiendo que se ingreso correctamente un 'n'
	for(int y = -(n-1); y <= (n-1); y++){
		for(int x = 1; x <= n; i++){
			if(CONDICION){
				printf("*");
			} else {
				printf(" ");
			}
		}
		// Acabado el 'for' interno debemos imprimir un salto de linea para
		// indicar una variacion en la verical
		printf("\n");
	}
}

```

De la figura que esperamos generar notamos que las posiciones en las que se imprime el asterisco cumplen _|x| + |y| <= n_, por lo que bastaría modificar el código anterior para que finalmente se dibuje el triángulo en función a 'n':

```c
#include <stdio.h>
#include <stdlib.h>

int main(){
	int n;
	scanf("%d", &n);
	for(int y = -(n-1); y <= n-1; y++){
		for(int x = 1; x <= n; x++){
			if(abs(y) + abs(x) <= n){
				printf("*");
			} else {
				printf(" ");
			}
		}
		// Acabado el 'for' interno debemos imprimir un salto de linea para
		// indicar una variacion en la verical
		printf("\n");
	}
}
```
Ingresando 5:
```
terminal~
5
*
**
***
****
*****
****
***
**
*
```
