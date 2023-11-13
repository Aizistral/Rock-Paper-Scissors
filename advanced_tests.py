import unittest
import random
from RPS_game import play, mrugesh, abbey, quincy, kris
from RPS import player

# This file features advanced and move extensive tests. They were designed
# to be less predictable than the tests in test_module.py, so that knowledge
# of the exact sequence in which the bots play, and number of games played
# by each bot could not be used to cheat the tests.
#
# These tests demonstrate compliance of Obliterator with additional requirements
# that I have described in the README.md file.


class UnitTests(unittest.TestCase):
    print()
    print("Performing advanced tests...")
    print()

    # This test will select each of the four bots in random order, and pair
    # them against the player for anywhere from 256 to 1024 games. This process
    # will be repeated until at least 8000 games are played.
    # Winrates against each bot will be recorded separately. In order to pass
    # the test, the player must achieve at least 60% minimum winrate against
    # each bot in each matchup.
    def test_player_vs_legion(self):
        opponent_pool = []
        winrates = {"quincy": [], "mrugesh": [], "kris": [], "abbey": []}
        games_remaining = 8000

        while games_remaining > 0:
            if len(opponent_pool) == 0:
                opponent_pool = [quincy, mrugesh, kris, abbey]

            # make sure every bot gets a turn
            random.shuffle(opponent_pool)
            opponent = opponent_pool.pop()

            games_to_play = random.randint(256, 1024)

            print(
                f"Testing the player against {opponent.__name__} in {games_to_play}-game matchup..."
            )

            winrate = round(play(player, opponent, games_to_play), 2)
            winrates[opponent.__name__].append(winrate)
            games_remaining -= games_to_play

            print(f"Achieved winrate: {winrate}%")
            print()

        print()
        print(f"Total games played: {8000 - games_remaining}")
        print()

        for opponent in winrates:
            average_winrate = (
                round(sum(winrates[opponent]) / len(winrates[opponent]), 2)
                if len(winrates[opponent]) > 0
                else 0
            )

            print(
                f"Winrates against {opponent}: avg {average_winrate}%, {winrates[opponent]}"
            )
            self.assertTrue(
                all(winrate >= 60 for winrate in winrates[opponent]),
                f"Failed to achieve 60% minimum winrate against {opponent}",
            )


if __name__ == "__main__":
    unittest.main()
