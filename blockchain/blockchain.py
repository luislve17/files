import hashlib
import json
from time import time
from time import sleep
from urllib.parse import urlparse
from uuid import uuid4

class Blockchain:
    def __init__(self):
        self.transaccion_actual = []
        self.cadena = []
        self.nodos = set()

        # Crea el bloque genesis
        self.nuevo_bloque(anterior_hash='1', prueba=100)

    def registrar_nodo(self, direccion):
        """
        Agrega un nuevo nodo a la lista de nodos
        :direccion: Direccion del nodo. ('http://192.168.0.5:5000')
        """

        parsed_url = urlparse(direccion)
        self.nodos.add(parsed_url.netloc)

    def validar_cadena(self, cadena):
        """
        Determina si el blockchain ddo es valido
        :cadena: Una blockchain
        :retorna: True si es valido, False si no lo es
        """

        ultimo_bloque = cadena[0]
        indice_actual = 1

        while indice_actual < len(cadena):
            bloque = cadena[indice_actual]
            print('{}'.format(ultimo_bloque))
            print('{}'.format(bloque))
            print("\n-----------\n")
            # Ve que el hash del bloque es correcto
            if bloque['anterior_hash'] != self.hash(ultimo_bloque):
                return False

            # Ve que la prueba de trabajo es correcta
            if not self.validar_prueba(ultimo_bloque['prueba'], bloque['prueba']):
                return False

            ultimo_bloque = bloque
            indice_actual += 1

        return True

    def resolver_conflictos(self):
        """
        Este es el algoritmo de consenso, resuelve conflictos
        reemplazandolo por la cadena de mayor longitud en la red.
        :retorna: True si nuestra cadena fue reemplaza, False si no lo fue
        """

        vecinos = self.nodos
        nueva_cadena = None

        # Buscamos solo cadenas con mayor longitud a la nuestra
        max_longitud = len(self.cadena)

        # Cogemos y verificamos las cadenas de todos los nodos de la red
        for nodo in vecinos:
            response = requests.get("http://{}/cadena".format(nodo))

            if response.status_code == 200:
                longitud = response.json()['longitud']
                cadena = response.json()['cadena']

                # Ve si la longitud es mayor y valida la cadena
                if longitud > max_longitud and self.validar_cadena(cadena):
                    max_longitud = longitud
                    nueva_cadena = cadena

        # Reemplaza nuestra cadena si descubre una de mayor longitud valida
        if nueva_cadena:
            self.cadena = nueva_cadena
            return True

        return False

    def nuevo_bloque(self, prueba, anterior_hash):
        """
        Crea un nuevo bloque en la Blockchain
        :prueba: 'prueba' o nounce dado por el algoritmo de prueba de trabajo
        :anterior_hash: Has del bloque anterior
        :retorna: Nuevo bloque
        """

        bloque = {
            'indice': len(self.cadena) + 1,
            'timestamp': time(),
            'transacciones': self.transaccion_actual,
            'prueba': prueba,
            'anterior_hash': anterior_hash or self.hash(self.cadena[-1]),
        }

        # Reseea la actual lista de transacciones
        self.transaccion_actual = []

        self.cadena.append(bloque)
        return bloque

    def nueva_transaccion(self, votante, candidato, monto):
        """
        Crea una nueva transaccion que ira en siguiente bloque a minar
        :votante: Direccion del Votante
        :candidato: Direccion del Candidato
        :monto: Monto
        :retorna: El indice del bloque que tendra esta transaccion
        """
        self.transaccion_actual.append({
            'votante': votante,
            'candidato': candidato,
            'monto': monto,
        })

        return self.ultimo_bloque['indice'] + 1

    @property
    def ultimo_bloque(self):
        return self.cadena[-1]

    @staticmethod
    def hash(bloque):
        """
        Crea un hash SHA-256  de un bloque
        :bloque: Bloque
        """

        # Debemos asegurarnos que el Diccionario este ordenado, o habra inconsistencias
        bloque_string = json.dumps(bloque, sort_keys=True).encode()
        return hashlib.sha256(bloque_string).hexdigest()

    def prueba_de_trabajo(self, ultima_prueba):
        """
        Algoritmo de prueba de trabajo:
         - Encuentra el numero p' tal que hash(pp') contiene adelante 4 ceros, donde p es el anterior p'
         - p es el anterior 'pruebba', y p' es el nuevo 'prueba'
        """

        prueba = 0
        while self.validar_prueba(ultima_prueba, prueba) is False:
            prueba += 1

        return prueba

    @staticmethod
    def validar_prueba(ultima_prueba, prueba):
        """
        Valia la 'prueba'
        :ultima_prueba: Anterior 'prueba'
        :prueba: Actual 'prueba'
        :retorna: True si es correcto, False si no lo es.
        """

        suposicion = '{}{}'.format(ultima_prueba,prueba).encode()
        suposicion_hash = hashlib.sha256(suposicion).hexdigest()
        return suposicion_hash[:4] == "0000"

