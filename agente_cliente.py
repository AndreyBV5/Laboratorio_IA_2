import random
import numpy as np
from flask import Flask, request, jsonify

# Entorno del juego
class CantStopEnvironment:
    def __init__(self):
        self.columns = {i: 0 for i in range(2, 13)}
        self.active_columns = []
        self.finished_columns = set()
        self.tienda_de_campaña = set()
        self.max_position = {2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3}
    
    def roll_dice(self):
        return [random.randint(1, 6) for _ in range(4)]
    
    def get_possible_moves(self, dice_roll):
        return [(dice_roll[i] + dice_roll[j], dice_roll[k] + dice_roll[l]) 
                for i, j, k, l in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]]
    
    def is_move_valid(self, move):
        return all(column in self.columns for column in move)
    
    def apply_move(self, move):
        for column in move:
            if column in self.active_columns or len(self.active_columns) < 3:
                self.active_columns.append(column)
                self.columns[column] += 1
                if self.columns[column] >= self.max_position[column]:
                    self.finished_columns.add(column)
                    self.active_columns.remove(column)
    
    def finalize_turn(self):
        self.tienda_de_campaña.update(self.active_columns)
        self.active_columns.clear()
    
    def reset_turn(self):
        for column in self.active_columns:
            self.columns[column] = max(self.columns[column] - 1, 0)
        self.active_columns.clear()
    
    def check_win(self):
        return len(self.finished_columns) >= 3
    
    def get_state(self):
        return tuple(sorted(self.columns.items())), tuple(sorted(self.active_columns))

# Agente Q-learning
class QLearningAgent:
    def __init__(self, environment, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.environment = environment
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, state, possible_moves):
        if np.random.rand() < self.epsilon:
            return random.choice(possible_moves)
        q_values = [self.q_table.get((state, move), 0) for move in possible_moves]
        max_q = max(q_values)
        return possible_moves[q_values.index(max_q)]
    
    def update_q_value(self, state, action, reward, next_state):
        old_value = self.q_table.get((state, action), 0)
        next_max = max([self.q_table.get((next_state, a), 0) for a in self.environment.get_possible_moves(self.environment.roll_dice())], default=0)
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[(state, action)] = new_value
    
    def play_turn(self):
        state = self.environment.get_state()
        dice_roll = self.environment.roll_dice()
        possible_moves = self.environment.get_possible_moves(dice_roll)
        action = self.choose_action(state, possible_moves)
        
        if self.environment.is_move_valid(action):
            self.environment.apply_move(action)
            if self.environment.check_win():
                self.update_q_value(state, action, 1, state)
                return True
            self.update_q_value(state, action, 0, state)
        else:
            self.environment.reset_turn()
            return False
        return None

# Configuración del servidor Flask
app = Flask(__name__)
environment = CantStopEnvironment()
agent = QLearningAgent(environment)

@app.route('/roll_dice', methods=['GET'])
def roll_dice():
    dice_roll = environment.roll_dice()
    return jsonify({'dice_roll': dice_roll})

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    move = tuple(data['move'])
    valid = environment.is_move_valid(move)
    if valid:
        environment.apply_move(move)
        win = environment.check_win()
        if win:
            return jsonify({'status': 'win'})
        return jsonify({'status': 'continue'})
    else:
        environment.reset_turn()
        return jsonify({'status': 'invalid'})

@app.route('/end_turn', methods=['POST'])
def end_turn():
    environment.finalize_turn()
    return jsonify({'status': 'turn ended'})

if __name__ == '__main__':
    app.run(debug=True)

