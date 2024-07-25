from flask import Flask, request, render_template, jsonify
import pandas as pd
import logging
import joblib

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar los modelos entrenados
model_regressor = joblib.load('Desercion_telesecundaria763.pkl')
scaler = joblib.load('scaler_desercion.pkl')
app.logger.debug('Modelo de clasificación y escalador cargados correctamente.')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        Trimestre3 = request.form['Trimestre3']
        Lengua_materna = request.form['Lengua_materna']
        Ocupacion_Jefe_Fam = request.form['Ocupacion_Jefe_Fam']
        Habitantes_Hogar = request.form['Habitantes_Hogar']
        Hermanos = request.form['Hermanos']
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[Trimestre3, Lengua_materna, Ocupacion_Jefe_Fam, Habitantes_Hogar, Hermanos]], 
                               columns=['Trimestre3', 'Lengua_materna', 'Ocupacion_Jefe_Fam', 'Habitantes_Hogar', 'Hermanos'])
        app.logger.debug(f'DataFrame creado: {data_df}')

        # Escalar los datos usando el escalador previamente guardado
        scaled_data = scaler.transform(data_df)
        app.logger.debug(f'Datos escalados: {scaled_data}')

        # Seleccionar y usar el modelo adecuado
        prediction = model_regressor.predict(scaled_data)
        
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Devolver las predicciones como respuesta JSON
        return jsonify({'probabilidad_desercion': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en el procesamiento: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
