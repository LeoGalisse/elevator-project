from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)

# Define fuzzy variables
error = ctrl.Antecedent(np.arange(-10, 10.1, 0.1), 'error')
delta_error = ctrl.Antecedent(np.arange(-10, 10.1, 0.1), 'delta_error')
p_motor = ctrl.Consequent(np.arange(0, 100.1, 0.1), 'p_motor')

# Define membership functions for error
error['negative'] = fuzz.trimf(error.universe, [-10, -10, 0])
error['zero'] = fuzz.trimf(error.universe, [-10, 0, 10])
error['positive'] = fuzz.trimf(error.universe, [0, 10, 10])

# Define membership functions for delta_error
delta_error['negative'] = fuzz.trimf(delta_error.universe, [-10, -10, 0])
delta_error['zero'] = fuzz.trimf(delta_error.universe, [-10, 0, 10])
delta_error['positive'] = fuzz.trimf(delta_error.universe, [0, 10, 10])

# Define membership functions for p_motor
p_motor['low'] = fuzz.trimf(p_motor.universe, [0, 0, 50])
p_motor['medium'] = fuzz.trimf(p_motor.universe, [0, 50, 100])
p_motor['high'] = fuzz.trimf(p_motor.universe, [50, 100, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(error['negative'] & delta_error['negative'], p_motor['low'])
rule2 = ctrl.Rule(error['negative'] & delta_error['zero'], p_motor['medium'])
rule3 = ctrl.Rule(error['negative'] & delta_error['positive'], p_motor['high'])
rule4 = ctrl.Rule(error['zero'] & delta_error['negative'], p_motor['low'])
rule5 = ctrl.Rule(error['zero'] & delta_error['zero'], p_motor['medium'])
rule6 = ctrl.Rule(error['zero'] & delta_error['positive'], p_motor['high'])
rule7 = ctrl.Rule(error['positive'] & delta_error['negative'], p_motor['low'])
rule8 = ctrl.Rule(error['positive'] & delta_error['zero'], p_motor['medium'])
rule9 = ctrl.Rule(error['positive'] & delta_error['positive'], p_motor['high'])

# Control system
elevator_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
elevator = ctrl.ControlSystemSimulation(elevator_ctrl)

# Function to update the elevator position
def update_position(current_position, e, de):
    elevator.input['error'] = e
    elevator.input['delta_error'] = de
    elevator.compute()
    power = elevator.output['p_motor']
    new_position = current_position * 0.996 + power * 0.00951
    return new_position

# Flask endpoint to handle control requests
@app.route('/control', methods=['POST'])
def control():
    data = request.json
    current_position = data['body']['current_position']
    desired_position = data['body']['desired_position']
    previous_error = data['body']['previous_error']
    
    e = desired_position - current_position
    de = e - previous_error
    new_position = update_position(current_position, e, de)
    previous_error = e
    
    response = {
        'current_position': new_position,
        'previous_error': previous_error
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
