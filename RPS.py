import random
import json
import numpy as np
import tensorflow as tf
from utils import GameHistory, move_to_int, int_to_move


# This class represents the destroyer of all Abbeys, Krises, Mrugeshes and Quincies,
# the one and only Obliterator. It is a neural network trained on over a million
# games played against the aforementioned bots.
class Obliterator:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)

    # We call this function to predict the best move based on the game history
    def make_move(self, game_history: GameHistory):
        # First, convert the game history to a model input
        model_input = self.__history_to_model_input(game_history)

        # Then, feed the input to the model
        obliterator_prediction = self.model(np.array([model_input]))

        # Finally, convert the prediction to a move
        predicted_move = np.argmax(obliterator_prediction)
        predicted_move = int_to_move(predicted_move)

        return predicted_move

    # Since the model expects a 128-move history represented as a list of
    # integers, we need to convert our game history to that format.
    # We will select the last 128 moves, or 64 game turns
    def __history_to_model_input(self, game_history: GameHistory):
        history = None

        # If history is even, it means that the last recorded move was opponent's.
        # Otherwise it was player's, in which case we need to remove it
        if (game_history.get_move_count() % 2) == 0:
            history = game_history.get_last_moves(128)
        else:
            history = game_history.get_last_moves(129)
            history.pop()

        # Map the moves to integers...
        history = list(map(lambda move: move_to_int(move), history))

        # Convert the history to numpy array...
        model_input = np.array(history)

        # Pad the input with random moves if it's less than 128 moves long
        if len(model_input) < 128:
            padding = np.random.randint(3, size=(128 - len(model_input)))
            model_input = np.concatenate((padding, model_input))

        # Our game history is a list of player_move -> opponent_move pairs.
        # We need to convert this to a list of opponent_move -> player_move pairs,
        # since this is what the model expects

        # First, split the input into pairs
        model_input = np.array(
            [model_input[i: i + 2] for i in range(0, len(model_input), 2)]
        )

        # Then, flip each pair
        model_input = np.flip(model_input, axis=1)

        # Finally, flatten the input
        model_input = model_input.flatten()

        # Aaaaand we're done
        return model_input


# I put the variables here because God did not intend default parameters to be used for persistence
game_history = GameHistory()
obliterator = Obliterator("./model/obliterator.keras")


# This function is responsible for teaching the enemy bots the true meaning of fear
def player(prev_play: str):
    if prev_play != "":
        game_history.add_opponent_move(prev_play)

    my_move = obliterator.make_move(game_history)
    game_history.add_player_move(my_move)

    return my_move
