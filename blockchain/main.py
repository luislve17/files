import hashlib
import json
# para grafico-------
from flask import Markup
from flask import render_template
# ------------------
import requests
from flask import Flask, jsonify, request, redirect, url_for
from flask import Flask, flash, redirect, render_template, request, session, abort
import os
# para manejo de usuarios
from loginManager import *
# -------------------
from blockchain import *

candidatos = ['Barnechea','Garcia','Guzman','Fujimori','PPK']
votos = [0,0,0,0,0]

# Instancia del Nodo
app = Flask(__name__)

# Genera una direccion unica para el nodo
identificador_nodo = str(uuid4()).replace('-', '')

# Instancia del Blockchain
blockchain = Blockchain()


@app.route('/minar', methods=['GET'])
def minar():
    # Probamos el algoritmo de prueba de trabajo para obtener la siguiente 'prueba'
    ultimo_bloque = blockchain.ultimo_bloque
    ultima_prueba = ultimo_bloque['prueba']
    prueba = blockchain.prueba_de_trabajo(ultima_prueba)

    # Se recibe un incentivo al encontrar la 'prueba'.
    # El votante es "0" para indicar que el nodo ha minado un bloque ganador.
    blockchain.nueva_transaccion(
        votante="0",
        candidato=identificador_nodo,
        monto=1,
    )

    # Agrega el nuevo bloque a la cadena
    anterior_hash = blockchain.hash(ultimo_bloque)
    bloque = blockchain.nuevo_bloque(prueba, anterior_hash)

    response = {
        'mensaje': "Bloque Nuevo Creado",
        'indice': bloque['indice'],
        'transacciones': bloque['transacciones'],
        'prueba': bloque['prueba'],
        'anterior_hash': bloque['anterior_hash'],
    }
    return jsonify(response), 200


@app.route('/transacciones/new', methods=['POST'])
def nueva_transaccion():
    valores = request.get_json()

    # Ve que los campos_requeridos
    campos_requeridos = ['votante', 'candidato', 'monto']
    if not all(k in valores for k in campos_requeridos):
        return 'Falta valores', 400

    # Crea una nueva transaccion
    indice = blockchain.nueva_transaccion(valores['votante'], valores['candidato'], valores['monto'])

    response = "mensaje: La transaccion sera agregada al Bloque {}".format(indice)
    return jsonify(response), 201


@app.route('/cadena', methods=['GET'])
def mostrar_cadena():
    response = {
        'cadena': blockchain.cadena,
        'longitud': len(blockchain.cadena),
    }
    return jsonify(response), 200


@app.route('/nodos/registrar', methods=['POST'])
def registrar_nodos():
    valores = request.get_json()

    nodos = valores.get('nodos')
    if nodos is None:
        return "Error: Proporcione una lista valida de nodos", 400

    for nodo in nodos:
        blockchain.registrar_nodo(nodo)

    response = {
        'mensaje': 'Nuevos nodos han sido agregados',
        'nodos_totales': list(blockchain.nodos),
    }
    return jsonify(response), 201

@app.route('/resultado/<string:nombre>', methods=['GET'])
def consulta(nombre):
    if not session.get('logged_in'):
        return render_template('loginScreen.html')
    direccion = nombre
    monto = 0

    for bloque in blockchain.cadena:
    	for transacciones in bloque["transacciones"]:
    		if direccion == transacciones["votante"]:#modificar por "candidato"
    			monto = monto + transacciones["monto"]
    			#blockchain.cadena[1]["transacciones"][0]["candidato"]
    response = {
        'mensaje': 'El monto total de la cuenta',
        'total': monto,
    }
    return jsonify(response), 201

@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('loginScreen.html')
    else:
        #flash("Usted ya voto")
        return render_template('layout.html')

@app.route('/login', methods=['POST'])
def login():
    usr_name = request.form['usuario.nombre']
    usr_dni = request.form['usuario.dni']

    if votacion_login(usr_name, usr_dni) == 1:
        session['logged_in'] = True
        session['user_id'] = usr_dni
        return redirect(url_for('votar'))
    else:
        flash('DNI o nombre de ciudadano incorrecto')
    return home()

@app.route("/logout")
def logout():
    session['logged_in'] = False
    session['user_id'] = None
    flash('Usted cerro sesión correctamente')
    return redirect('/')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/votar', methods=['GET','POST'])
def votar():
    if not session.get('logged_in'):
        flash("Necesita logearse primero")
        return redirect('/')

    votante_dni = session['user_id']
    user = get_user(votante_dni)

    if user['credito'] == 0:
        flash("Usted ya votó")
        return redirect('/')

    votante = user['nombre']
    if request.method == 'GET':
        return render_template("formulario.html",usr_name = votante,usr_dni=votante_dni)
    else:
        voto = request.form['opc']
        indice_temp = candidatos.index(voto)
        votos[indice_temp]=votos[indice_temp]+1
        indice = blockchain.nueva_transaccion(votante,voto, 1)
        votacion_confirm(votante_dni)

        response = "mensaje: Voto agregado al bloque {}".format(indice)
        user['credito'] = 0
        minar()
        return render_template("notif_2.html"), 201

@app.route('/nodos/resolver', methods=['GET'])
def consenso():
    reemplazado = blockchain.resolver_conflictos()

    if reemplazado:
        response = {
            'mensaje': 'Nuestra cadena fue reemplazada',
            'nueva_cadena': blockchain.cadena
        }
    else:
        response = {
            'mensaje': 'Nuestra cadena es predominante',
            'cadena': blockchain.cadena
        }

    return jsonify(response), 200

@app.route('/chartBar', methods = ['POST', 'GET'])
def chartBar():
    labels = candidatos
    values = votos
    return render_template('chartBar.html', values=values, labels=labels)

@app.route("/chartPie")
def chartPie():
    labels = candidatos
    values = votos
    colors = [ "#F7464A", "#46BFBD", "#FDB45C","#FEDCBA","#ABCDEF"]
    return render_template('chartPie.html', set=zip(values, labels, colors))




if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app.run(debug=True,host='0.0.0.0', port=port)
