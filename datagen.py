import random
import json
from bots import Bot, Randomazzo, Quincy, Mrugesh, Kris, Abbey
from utils import GameHistory, int_to_move, move_to_int, ideal_response, ideal_response_int


class DataGen:
    def __init__(self):
        self.quincy = Quincy()
        self.mrugesh = Mrugesh()
        self.kris = Kris()
        self.abbey = Abbey()
        self.random_player = Randomazzo()

        self.opponent_pool = [self.quincy, self.mrugesh, self.kris, self.abbey]
        self.game_history = GameHistory(switched_order=True)
        self.data = []

    def generate_matchup_data(self, sample_size=1000, visual=False, verbose=False):
        for i in range(sample_size):
            if i % 100 == 0:
                print("Generating data: ", i, "/", sample_size)

            self.__reset_histories()

            opponent = random.choice(self.opponent_pool)

            if random.random() < 0.5:
                opponent = self.abbey  # need more training data for abbey

            random_games = random.randint(70, 256)

            self.__play(
                self.random_player,
                opponent,
                random_games,
                record=True,
                override=False,
                visual=visual,
            )

            # Play from some more games with override enabled.
            # override will pick an ideal response to bot's play some % of the time

            override_games = random.randint(0, 256 + 64) - 64

            if override_games > 0:
                self.__play(
                    self.random_player,
                    opponent,
                    override_games,
                    record=True,
                    override=True,
                    visual=visual,
                    verbose=verbose,
                )

            # Get the last 130 moves from the game history
            last_130 = self.game_history.get_last_moves(130)

            # Use the first 128 elements as the batch
            batch = last_130[:128]

            # Use the last element to determine the label, based on what would
            # have been the winning move for the player against next opponent move
            label = last_130[128]

            if visual:
                label = ideal_response(label[-1])
            else:
                label = ideal_response_int(label)

            self.data.append((label, batch))

        print(f"Generated {len(self.data)} data points.")

        return self.data

    def save_data(self, file_name="training_data.json"):
        print("Saving data to", file_name)
        with open(file_name, "w") as f:
            json.dump(self.data, f)

    def load_data(self, file_name="training_data.json"):
        print("Loading data from", file_name)
        with open(file_name, "r") as f:
            self.data = json.load(f)
            return self.data

    # This function is copied from RPS_game.py, with some modifications
    # to record game history and produce more varied games
    def __play(self, player1: Bot, player2: Bot, num_games: int, verbose=False, record=False,
               visual=False, override=False, showResult=False):
        p1_prev_play = ""
        p2_prev_play = ""
        results = {"p1": 0, "p2": 0, "tie": 0}

        for i in range(num_games):
            p1_play = player1.make_move(p2_prev_play)
            p2_play = player2.make_move(p1_prev_play)

            if override:
                # pick an ideal response to p2_play some % of the time
                # otherwise, pick one of 2 other options randomly

                if random.random() < 0.75:
                    p1_play = ideal_response(p2_play)
                else:
                    not_ideal_responses = ["R", "P", "S"]
                    not_ideal_responses.remove(ideal_response(p2_play))

                    p1_play = random.choice(not_ideal_responses)

            if p1_play == p2_play:
                results["tie"] += 1
                winner = "Tie."
            elif (
                (p1_play == "P" and p2_play == "R")
                or (p1_play == "R" and p2_play == "S")
                or (p1_play == "S" and p2_play == "P")
            ):
                results["p1"] += 1
                winner = "Player 1 wins."
            elif (
                p2_play == "P"
                and p1_play == "R"
                or p2_play == "R"
                and p1_play == "S"
                or p2_play == "S"
                and p1_play == "P"
            ):
                results["p2"] += 1
                winner = "Player 2 wins."

            if record:
                if visual:
                    self.game_history.add_opponent_move("O_" + p2_play)
                    self.game_history.add_player_move("P_" + p1_play)
                else:
                    self.game_history.add_opponent_move(move_to_int(p2_play))
                    self.game_history.add_player_move(move_to_int(p1_play))

            if verbose:
                print("Player 1:", p1_play, "| Player 2:", p2_play)
                print(winner)
                print()

            p1_prev_play = p1_play
            p2_prev_play = p2_play

        games_won = results["p2"] + results["p1"]

        if games_won == 0:
            win_rate = 0
        else:
            win_rate = results["p1"] / games_won * 100

        if showResult:
            print("Final results:", results)
            print(f"Player 1 win rate: {win_rate}%")

        return win_rate

    def __reset_histories(self):
        for bot in self.opponent_pool:
            bot.reset()

        self.random_player.reset()
        self.game_history.clear()

