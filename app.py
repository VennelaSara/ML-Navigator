import joblib
from flask import Flask, request, jsonify, render_template
import pandas as pd
from your_file_name import recommend_route, preprocess_traffic, preprocess_seattle_weather, preprocess_road_accidents

app = Flask(__name__)

# Load datasets and preprocess them
road_accidents = pd.read_csv('Road_Accidents_2017-Annuxure_Tables_4.csv')
traffic = pd.read_csv('traffic.csv')
seattle_weather = pd.read_csv('seattle-weather.csv')

road_accidents = preprocess_road_accidents(road_accidents)
traffic = preprocess_traffic(traffic)
seattle_weather = preprocess_seattle_weather(seattle_weather)

# Load trained model
model = joblib.load('your_model_filename.pkl')  # Adjust the path as needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_preferences = {
        'avoid_rain': bool(request.form['avoid_rain']),
        'preferred_temp_range': (float(request.form['min_temp']), float(request.form['max_temp'])),
        'avoid_high_traffic': bool(request.form['avoid_high_traffic'])
    }

    recommendations = recommend_route(user_preferences, traffic, seattle_weather, model)

    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
