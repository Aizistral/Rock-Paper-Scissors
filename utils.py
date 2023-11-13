import random


# Map moves to integers
def move_to_int(move: str):
    return {"R": 0, "P": 1, "S": 2}[move]


# Map integers to moves
def int_to_move(move: int):
    return {0: "R", 1: "P", 2: "S"}[move]


# Return the move that beats the given move
def ideal_response(move: str):
    return {"R": "P", "P": "S", "S": "R"}[move]


# Return the move that beats the given move, as an integer
def ideal_response_int(move: int):
    return {0: 1, 1: 2, 2: 0}[move]


# A simple class to keep track of the game history.
# Both player and opponent moves are recorded, and by default history
# must start with a player move.
class GameHistory:
    def __init__(self, switched_order=False):
        self.history = []
        self.switched_order = switched_order
        self.last_move_player = self.switched_order

    def add_player_move(self, move: str | int):
        if self.last_move_player:  # missed opponent's move?
            self.history.pop()

        self.history.append(move)
        self.last_move_player = True

    def add_opponent_move(self, move: str | int):
        if not self.last_move_player:  # missed player's move?
            self.history.pop()

        self.history.append(move)
        self.last_move_player = False

    def get_last_moves(self, move_amount: int):
        return self.history[-move_amount:]

    def get_last_move(self) -> str | int:
        return self.history[-1]

    def get_move_count(self):
        return len(self.history)

    def clear(self):
        self.history = []
        self.last_move_player = self.switched_order
