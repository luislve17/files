import hashlib
import sys

database = [
	{'nombre' : 'Josias Ruegg Yupa', 'dni' : '77797353', 'dni_hash' : '1', 'credito' : 1},
	{'nombre' : 'Luis Vasquez Espinoza', 'dni' : '73641449', 'dni_hash' : '2', 'credito' : 1},
	{'nombre' : 'Diego Cabanillas Coaquira', 'dni' : '48445894', 'dni_hash' : '3', 'credito' : 1},
	{'nombre' : 'Pedro', 'dni' : '123', 'dni_hash' : '4', 'credito' : 1},
	{'nombre' : 'Juan', 'dni' : '321', 'dni_hash' : '5', 'credito' : 1}
]
dni_hash = 0

def votacion_login(name, dni):
	global database
	valid_user = -1
	try:
		suspect = next(item for item in database if item['dni'] == dni)
		if suspect['nombre'] == name:
			valid_user = 1
	except StopIteration:
		valid_user = -1
	return valid_user

def get_user(dni):
	try:
		suspect = next(item for item in database if item['dni'] == dni)
		return suspect
	except StopIteration:
		return None

def votacion_confirm(dni):
	global database
	global dni_hash
	database.append(dni_hash)
