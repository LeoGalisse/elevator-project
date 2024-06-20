from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)

# Define fuzzy variables
error = ctrl.Antecedent(np.arange(0, 25.01, 0.01), 'error')
delta_error = ctrl.Antecedent(np.arange(-50, 50.1, 0.1), 'delta_error')
p_motor = ctrl.Consequent(np.arange(0, 90.5, 0.5), 'p_motor')

# Define membership functions for error
error['zero'] = fuzz.trapmf(error.universe, [0, 0, 0.1, 0.2])
error['d1'] = fuzz.trimf(error.universe, [0.15, 0.2, 9])
error['d2'] = fuzz.trimf(error.universe, [0.2, 9, 15])
error['d3'] = fuzz.trimf(error.universe, [9, 15, 21])
error['d4'] = fuzz.trimf(error.universe, [15, 21, 25])

# Define membership functions for delta_error
delta_error['much_negative'] = fuzz.trapmf(delta_error.universe, [-50, -50, -24, -12])
delta_error['less_negative'] = fuzz.trimf(delta_error.universe, [-24, -0.2, 0])
delta_error['zero'] = fuzz.trimf(delta_error.universe, [-0.2, 0, 0.2])
delta_error['less_positive'] = fuzz.trimf(delta_error.universe, [0, 0.2, 24])
delta_error['much_positive'] = fuzz.trapmf(delta_error.universe, [12, 24, 50, 50])

# Define membership functions for p_motor
p_motor['initial'] = fuzz.trimf(p_motor.universe, [0, 0, 31.5])
p_motor['low'] = fuzz.trimf(p_motor.universe, [0, 31.5, 45])
p_motor['medium'] = fuzz.trimf(p_motor.universe, [31.5, 45, 90])
p_motor['high'] = fuzz.trimf(p_motor.universe, [45, 90, 90])

# Define fuzzy rules
rule1 = ctrl.Rule(error['zero'] & delta_error['much_negative'], p_motor['initial'])
rule2 = ctrl.Rule(error['zero'] & delta_error['less_negative'], p_motor['initial'])
rule3 = ctrl.Rule(error['zero'] & delta_error['zero'], p_motor['initial'])
rule4 = ctrl.Rule(error['zero'] & delta_error['less_positive'], p_motor['initial'])
rule5 = ctrl.Rule(error['zero'] & delta_error['much_positive'], p_motor['initial'])
rule6 = ctrl.Rule(error['d1'] & delta_error['much_negative'], p_motor['low'])
rule7 = ctrl.Rule(error['d1'] & delta_error['less_negative'], p_motor['low'])
rule8 = ctrl.Rule(error['d1'] & delta_error['zero'], p_motor['low'])
rule9 = ctrl.Rule(error['d1'] & delta_error['less_positive'], p_motor['low'])
rule10 = ctrl.Rule(error['d1'] & delta_error['much_positive'], p_motor['low'])
rule11 = ctrl.Rule(error['d2'] & delta_error['much_negative'], p_motor['medium'])
rule12 = ctrl.Rule(error['d2'] & delta_error['less_negative'], p_motor['medium'])
rule13 = ctrl.Rule(error['d2'] & delta_error['zero'], p_motor['medium'])
rule14 = ctrl.Rule(error['d2'] & delta_error['less_positive'], p_motor['medium'])
rule15 = ctrl.Rule(error['d2'] & delta_error['much_positive'], p_motor['medium'])
rule16 = ctrl.Rule(error['d3'] & delta_error['much_negative'], p_motor['high'])
rule17 = ctrl.Rule(error['d3'] & delta_error['less_negative'], p_motor['high'])
rule18 = ctrl.Rule(error['d3'] & delta_error['zero'], p_motor['high'])
rule19 = ctrl.Rule(error['d3'] & delta_error['less_positive'], p_motor['high'])
rule20 = ctrl.Rule(error['d3'] & delta_error['much_positive'], p_motor['high'])
rule21 = ctrl.Rule(error['d4'] & delta_error['much_negative'], p_motor['high'])
rule22 = ctrl.Rule(error['d4'] & delta_error['less_negative'], p_motor['high'])
rule23 = ctrl.Rule(error['d4'] & delta_error['zero'], p_motor['high'])
rule24 = ctrl.Rule(error['d4'] & delta_error['less_positive'], p_motor['high'])
rule25 = ctrl.Rule(error['d4'] & delta_error['much_positive'], p_motor['high'])

# Control system
elevator_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25])
elevator = ctrl.ControlSystemSimulation(elevator_ctrl)

# Function to update the elevator position
def update_position(current_position, desired_position, e, de):
    print(f'Current position: {current_position}, Error: {e}, Delta error: {de}')
    elevator.input['error'] = e
    elevator.input['delta_error'] = de
    elevator.compute()
    position, _ = [[current_position], current_position]
    for time in np.arange(0.1, 3.1, 0.1):
        power = time * 0.315 / 3
        print(f'Power: {power}')
        new_position = current_position * 0.996 * (1 if e >= 0 else -1) + power * 0.00951
        print(f'New position: {new_position}')
        if time == round(time, 0):
            position = np.append(position, new_position)
        
    previous_error = desired_position - new_position
    
    for _ in np.arange(3, 400, 1):
        e = desired_position - new_position
        current_error = abs(e)
        elevator.input[error.label] = current_error
        
        current_delta_error = previous_error - current_error
        elevator.input[delta_error.label] = current_delta_error
        
        elevator.compute()
        power = elevator.output['p_motor']
        print(f'Power: {power}')
        new_position = abs(new_position * 0.996 * (1 if e >= 0 else -1) + power * 0.00951)
        
        position = np.append(position, new_position)
        previous_error = current_error
        
    if (desired_position > current_position):
        ymin, ymax = [current_position - 3, max(position) * 1.25]
    else:
        ymin, ymax = [min(position) * 0.75, current_position + 3]
        
    print(f'ymin: {ymin}')
    print(f'ymax: {ymax}')
        
    return new_position

# Flask endpoint to handle control requests
@app.route('/control', methods=['POST'])
def control():
    data = request.json
    current_position = data['body']['current_position']
    desired_position = data['body']['desired_position']
    previous_error = data['body']['previous_error']
    
    print(f'Current position: {current_position}, Desired position: {desired_position}, Previous error: {previous_error}')
    e = desired_position - current_position
    de = e - previous_error
    new_position = update_position(current_position, desired_position, e, de)
    previous_error = e
    
    response = {
        'current_position': new_position,
        'previous_error': previous_error
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
