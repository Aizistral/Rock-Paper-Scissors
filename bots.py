import random

# Here I implement the bots from RPS_game.py using classes.
# This is done to allow resetting their state between matchups as needed,
# in particular for generation of training data.


class Bot:
    def make_move(self, prev_opponent_play):
        raise NotImplementedError

    def reset(self):
        pass

    def __str__(self):
        return self.__class__.__name__


class Randomazzo(Bot):
    def make_move(self, prev_opponent_play):
        return random.choice(["R", "P", "S"])


class Quincy(Bot):
    def __init__(self):
        self.counter = 0
        self.choices = ["R", "R", "P", "P", "S"]

    def make_move(self, prev_opponent_play):
        self.counter += 1
        return self.choices[self.counter % len(self.choices)]

    def reset(self):
        self.counter = 0


class Mrugesh(Bot):
    def __init__(self):
        self.opponent_history = []

    def make_move(self, prev_opponent_play):
        self.opponent_history.append(prev_opponent_play)
        last_ten = self.opponent_history[-10:]
        most_frequent = max(set(last_ten), key=last_ten.count)

        if most_frequent == "":
            most_frequent = "S"

        ideal_response = {"P": "S", "R": "P", "S": "R"}
        return ideal_response[most_frequent]

    def reset(self):
        self.opponent_history = []


class Kris(Bot):
    def make_move(self, prev_opponent_play):
        if prev_opponent_play == "":
            prev_opponent_play = "R"

        ideal_response = {"P": "S", "R": "P", "S": "R"}

        return ideal_response[prev_opponent_play]


class Abbey(Bot):
    def __init__(self):
        self.opponent_history = []
        self.play_order = [
            {
                "RR": 0,
                "RP": 0,
                "RS": 0,
                "PR": 0,
                "PP": 0,
                "PS": 0,
                "SR": 0,
                "SP": 0,
                "SS": 0,
            }
        ]

    def make_move(self, prev_opponent_play):
        if not prev_opponent_play:
            prev_opponent_play = "R"

        self.opponent_history.append(prev_opponent_play)

        last_two = "".join(self.opponent_history[-2:])

        if len(last_two) == 2:
            self.play_order[0][last_two] += 1

        potential_plays = [
            prev_opponent_play + "R",
            prev_opponent_play + "P",
            prev_opponent_play + "S",
        ]

        sub_order = {
            k: self.play_order[0][k] for k in potential_plays if k in self.play_order[0]
        }

        prediction = max(sub_order, key=sub_order.get)[-1:]

        ideal_response = {"P": "S", "R": "P", "S": "R"}
        return ideal_response[prediction]

    def reset(self):
        self.opponent_history = []
        self.play_order = [
            {
                "RR": 0,
                "RP": 0,
                "RS": 0,
                "PR": 0,
                "PP": 0,
                "PS": 0,
                "SR": 0,
                "SP": 0,
                "SS": 0,
            }
        ]
