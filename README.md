# Rock Paper Scissors

In front of you lies a project that consumed good 3 days of my life. Granted the complexity and specific details of my final solution, I feel like a little justification is required, so here we go.

## The Problem

You may familiarize yourself with project requirements provided by freeCodeCamp here: https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/rock-paper-scissors

First and foremost, I feel obliged to state that the scope of this particular assignment is *incredibly* unclear. It fails to make it evident which approach the assignee is expected to take, or answer the most natural questions about what they are or aren't allowed to do. For instance:

- Which libraries are we allowed to use, if any?

- Are we allowed to look at the code of the bots in `RPS_game.py`, and build our solution knowing exactly how they work?

- Are we even allowed to know the exact number and identities of opponent bots?

- Are we allowed to look at the test cases, and utilize the knowledge of how exactly they are structured?

- Should a specific machine learning algorithm be used? Can we use pre-trained models and/or external data?

The assignment doesn't even say where exactly machine learning is supposed to come into picture, or if it is indeed mandatory for it to come into picture at all. While other certification projects have a clear counterpart that was explored in the curriculum, there is no such thing for this one, leaving the unfortunate assignee completely lost.

I will admit that in the end, I enjoyed the challenging and open-ended nature of this assignment, and had a lot of fun experimenting with different approaches. However, I feel like it would simply be *too much* for most people, forcing them to either give up or resort to copying someone else's solution. This is especially bad considering that this is the first certification project on the list, and none of the other ones are even remotely as challenging.

## Assumptions

Since the assignment refused to answer my questions on what I can and cannot do, I had to make up the answers on my own.

**As such, I hereby decree:**

- I must only use libraries, machine learning algorithms, and data that were presented, used and explored in the Machine Learning with Python course.

- I must employ some kind of machine learning algorithm as part of my solution. There are no specific requirements on which one to use or how, only that it must be consistent with the requirement above.

- While I am prohibited from modifying `RPS_game.py`, I am allowed to look at its code to learn the exact number and identities of opponent bots, as well as every detail of their implementation.

- I am allowed to use the bots from `RPS_game.py` and reproduce their behavior in order to build my solution. In particular, I am allowed to use them to generate unlimited amounts of training data.

- I am NOT allowed to exploit the knowledge of the exact order of opponents, or exact number of games played against each opponent, in order to beat them. This is particularly relevant to test cases. In test cases, opponent bots always play in the same order, exactly 1000 games each. Utilizing this knowledge it would be trivial to manually build a perfect anti-bot for each opponent bot, and simply engage the appropriate anti-bot for every 1000 games. This would also make machine learning completely redundant.

- I am allowed to know that each opponent plays a reasonably large number of games to leave a chance to figure out their identity, and make the matchup results statistically significant. For the purposes of this project I decree that "reasonably large" is 256 or more. This number is chosen based on nothing but my gut feeling though 🤷‍♀️

- Finally, I am allowed to not interpret "write all your code in `RPS.py`" instruction literally, and organize my code into multiple files. This is nigh-required to keep things clean and readable, so yeah, there you go.

## Early Attempts

With those assumptions established, the fans of my brain-GPU have already started spinning. The first approach that crossed my mind can be summarized as follows:

1. I can still create a perfect anti-bot to counter each particular opponent bot. A surface-level glance at `RPS_game.py` leads me to believe that once I figure out which bot I am playing against, it should be easy to employ a perfect strategy against it and proceed to play all subsequent games against it with 100% winrate.

2. Then, all I have to do to achieve success is figure out opponent's identities. This is where machine learning comes into picture. I will create a model that will look at the history of last `n` games, and determine which enemy bot I am playing against. Then, I will use the appropriate anti-bot to make my move.

3. To train said model, I will play and record games against all opponent bots. Since I don't have a high-accuracy algorithm to beat them yet, I will simply make random moves against them. The model will have to know the history of both player and opponent moves to make a prediction, since some bots use player's moves to choose their own. Therefore, using random moves will have an additional benefit of preventing the model from trying to base its predictions off of unintended patterns in player's moves.

Should this approach work, it would turn the whole project into a classification problem, a type of problem reasonably well-explored and covered in the course. I was eager to start generating the training data and training the model at the heart of this whole operation. However, I decided to double-check one of my assumptions first, just to be safe.

That is, the assumption that I can easily and reliably counter each bot once I figure out its identity, even if it takes a few dozen consecutive games to do so. A deeper look at `RPS_game.py` almost dispelled my worries... almost:

- **Quincy:** plays the same 5-move sequence on repeat. Trivial to counter by simply knowing its previous move.

- **Mrugesh:** looks at the last 10 moves made by the player, and tries to play an ideal response to the most frequent move. A bit harder to counter, but simply keeping track of your own last 10 moves should be sufficient to "dodge" all of its attacks.

- **Kris:** plays a perfect response to player's previous move. Trivial to counter by remembering your own previous move.

- **Abbey:** okay, this one's bit complex. It keeps a full history of player's moves. Upon appending a newly received player move to this history, it will look at the last pair of moves made by the player, and add +1 to its entry in the "frequency table". This table contains all possible pairs of moves, of which there are 9. To make its own move, it will look at the last move made by the player, and create a sub-table of pairs that start with that move. It will then look at the most frequent pair in the table, and play an ideal response to the ending move of that pair.<br>In short, it will try to figure out the most popular "continuation" to the player's previous move, and counter it. Well, well, well, how do we defeat this?

What immediately struck me as odd is that unlike other bots, Abbey does not look at last `n` moves of opponent's history. It keeps track of the whole history at all times, never resetting.

Naturally, this is not an issue if we know exactly when we are playing against Abbey. Simply keep track of our own history of moves made specifically against this bot, use exactly the same algorithm it uses to determine its move, and counter that. It will always remain one step behind.

However, as per requirements established in the previous section, I am not allowed to just *know* that I am playing against Abbey. I have to figure it out based on the history of moves. More specifically, a machine learning model will figure it out for me.

The first concern here was actually how the model will do that in the first place. Since a model can only receive so much input data at once, the most I can give it is the last `n` moves of the game, not the whole history. If we consider that I intended to play random moves against each bot to generate training data, it is a natural assumption that Abbey's responses would look just as random if we only consider a small segment of move history. However, this concern was quickly dispelled by the fact that Abbey is the only bot of such complexity, so the network can learn to identify it by the process of elimination.

Still, the most naive design for anti-Abbey bot requires to have an exact copy of the information Abbey has, at all times. Since we cannot possibly achieve 100% identification accuracy, how do we determine which specific moves were made against Abbey, and which were against some other bot we might have played previously? If we only consider the moves we made when we were *confident* that we are playing against Abbey, our information will be inherently incomplete and imperfect, and there is no more guarantee that the anti-bot will achieve 100% winrate, or indeed any substantial winrate at all.

Well, it doesn't hurt to try anyways, so here goes the ye 'ol trusty empirical research. First, I coded my naive anti-Abbey bot, and verified that on clean start `play(antiabbey, abbey, 1000)` indeed yields 100% winrate for my champion anti-Abbey. Then, I simulated the unknowns by pairing Abbey against a bot that always plays a random move, right before the matchup against anti-Abbey: `play(random_player, abbey, 64)`. The purpose was to pre-fill Abbey's frequency table with inherently unaccessible information. Number 64 is a bit arbitrary here, but it's not unreasonable to assume that throughout the course of a 1000-game matchup in "field conditions", my yet-to-be-trained neural network will fail to identify Abbey at least that many times, especially in early games, where I could have been paired with a different bot before.

The results were... well, not particularly impressive. Anti-Abbey has failed to score anywhere near the minimum required winrate of 60%, and consistently fluctuated somewhere around 50%.

So, there goes that approach. Of course, I was unwilling to give up that easily, and tried a few other designs for potential anti-bot. They ranged from looking only at some last `n` games, to using Monte Carlo sampling to pick a limited subset of moves previously made against Abbey. All hope was that pair frequency deduced from a limited subset of moves could be sufficiently representative of actual frequency table at Abbey's disposal to beat the odds, even if not by much. Unfortunately, none of these approaches could yield the target winrate of 60%, and some failed spectacularly, at times scoring less than 10%.

## In Search of Inspiration

While at first I was resolved to design a solution fully on my own, and avoid looking at other people's solutions even for as much as a spark of inspiration, around this point my patience failed me. I thought that I must have missed something obvious, and am wasting time solely because of my own incompetence.

I searched up what other people had to say about this project, in hopes to quickly find the "obvious something" that I missed and go back to implementing my solution. I was surprised to find that the general consensus was that this project is indeed very hard, the Abbey bot is it's main problem, and that most people had to base their work on someone else's. I also found exactly zero solutions that involved a classification model such as mine, or indeed any form of machine learning.

In particular, I looked at this article by Sri Hartini at medium.com: https://medium.com/@sri.hartini/rock-paper-scissors-in-python-5173ab69ca7a

It is a wonderful article that showcases a lot of experimentation and research done by its author, who have endured a struggle not dissimilar to my own in completing this assignment.

The final solution that the author arrived at is essentially to implement Abbey v2.0, a bot based on the same basic logic as Abbey, but extended to monitor frequencies of sequences of moves longer than just pairs. The author says that increasing sequence length beyond 6 moves yields terrible results, so it is best kept in range from 3 to 6. Through their own experimentation they found 5 to be the most optimal. This bot turned out to be capable of beating not only the original Abbey bot, but also all other bots in `RPS_game.py`.

However, trying to experiment with their solution, I have found that the winrate of Abbey v2.0 vs Abbey is on average only *slightly* higher than 60%, and occasionally drops to as low as 52%. This depended on the amount of random games that I preloaded original Abbey with and, well, just luck. Even though Abbey v2.0 is technically deterministic if played against all opponent bots, and therefore passes the original test cases, this fails to satisfy my assumed requirement that knowledge of the exact order of opponents and amount of games played against them must not be depended upon. Furthermore, I would have to sacrifice the idea to use a classification model at all, therefore also failing to satisfy the requirement that machine learning technology must be employed in the project.

## A New Hope

Well, back to the drawing board then, aye? For a while, I kept brainstorming possible approaches to manually code an efficient anti-Abbey bot. It seemed inconcievable to me that there would be no mathematically simple and elegant way to achieve statistical advantage against it, given imperfect information. Why would it even be in this assignment if that was the case?

My efforts, however, remained in vain. It was then that I decided - well, if the task is too complex and tedious for a human brain, why not let a computer do it? This is a machine learning project after all, so I might as well train 2 models - one for classification, and one specifically to beat Abbey. From here on, I will refer to the latter as simply "Antibot", as that was what I called it in my code.

Training the Antibot would yield an additional benefit - if there indeed was a mathematically simple and computationally feasible way to predict Abbey with imperfect information, the model would quickly converge to it, and easily achieve winrates that are both very high and consistent.

Initially, I have decided that the Antibot will take in the history of last 32 games, contaning both player and opponent moves. Model input could therefore be represented as 64 integers taking on values of either 0, 1 or 2. The model will attempt to predict Abbey's next move following that history, so the last layer of the model will have 3 neurons, each representing a possible move.

To generate training data for the Antibot, I decided to take the following approach:

- I will play and and record from 128 to 256 random moves against Abbey. The purpose is to prefill Abbey's frequency table, simulating a state where we're deep into a matchup.

- To create a training sample, I will take 33 games from the tail of recorded history. The first 32 games will be used as input, and Abbey's move in a 33rd game will be the label.

- Rinse and repeat until I have enough training data. Naturally, I will reset Abbey's frequency table before each training sample is generated.

This data could then be dumped into a file and loaded back in for training/validation as needed.

Generating the data and training the Antibot is where I spent most of my time on this project, as much tweaking hyperparameters and fine-tuning the datagen, as just finding bugs in my own code. At the end, I extended model input from 32 to 64 games (or 128 integers), and increased the amount of training data used from a few thousand samples to over a million.

In my early attempts, the Antibot was easily able to achieve 85% prediction accuracy on validation data. It was a good result on the surface, and hypothetically - it should have translated into winrates even higher than 85%, since even if it doesn't predict the right move, there's still around 50/50 chance between a loss and a tie.

Unfortunately, once I tried to deploy the model into a real matchup to predict Abbey's moves and counter them, its prediction accuracy plummeted, fluctuating in 45-60% range. I double-checked that no bugs in my code are at fault for this, by trying to use the model throughout the matchup to make predictions, but not actually using those predictions to make moves, instead making the moves randomly. With this setup the accuracy spiked back up to expected 85%. To this time clear understanding of why that was the case still eludes me. My working theory is that it learned to predict patterns in the distribution of pseudo-random numbers generated by Python's `random` module, which would be absolutely fascinating, but unfortunately not very useful.

At this point I was compelled to use an entirely different machine learning technique. In particular, reinforcement learning seemed very promising for this task, as then my model would be able to learn dynamically from direct confrontation with Abbey, guaranteeing its efficiency in field conditions. Unfortunately, the machine learning course from FCC very briefly touches on the topic of reinforcement learning, mostly just explaining theory. The only practically demonstrated approach in the course is Q-Learning, which I believe is not suitable for this task. As I was resolved to complete the project using only the knowledge and techniques taught in the course, I had to abandon this idea.

To make any undesirable patterns less pronounced and hopefully better approximate a real matchup, I adjusted my data generation algorithm. First, it would still play and record anywhere from 128 to 256 games at random. After this, it would play and record another 64 to 256 games, in what I called "override" mode. Override mode essentially cheats against Abbey, and substitutes player's move with a perfect response to Abbey's move 75% of the time. In remaining 25% of cases, it will pick among two other possible moves, which will either lead to a tie or a loss.

A batch of 64 games (+1 for the label) will then be selected from a random point in the recorded history, so it could be all random games, all high-accuracy games, or a mix of both. The rest of the process remains the same.

The idea behind override mode was to approximate the gameplay of a skilled player, who knows some kind of strategy to beat Abbey, whatever it may be. The model should then have less trouble performing when it itself is used to represent said skilled player.

Right off the bat, with this kind of dataset it had much harder time achieving high accuracy during the training. It wasn't concerning, however, as even consistent 60% prediction accuracy would have been well enough to beat Abbey in a matchup, assuming it would actually translate into a real matchup (remember, expected winrate is higher than prediction accuracy).

After training for 200 epochs on ~0.5 gigabytes of data, which took over 2 hours on my machine, the model achieved about 75% prediction accuracy on validation data. I promptly paired my freshly baked Antibot against Abbey in a real confrontation, and... it worked! Kind of. Prediction accuracy still did not entirely translate, and now it fluctuated around 55%. However, the winrate firmly mounted itself around 68% mark, and my repeated tests never showed it drop below 65%. I determined that this is a passing grade, and was ready to call it a day.

## Birth of the Obliterator

While I was working on all that, another idea crossed my mind. Since I have to invoke a machine learning model every game anyways, wouldn't it make sense to train a model to directly beat all bots, not just Abbey? Then I wouldn't have to worry about classifying opponents, and my final code would be much simpler.

Initially I decided that such a thing would be beyond the paygrade of this project, and I should just focus on finishing the implementation of my original idea. However, once I was done with that, I decided to give it a shot anyways, just to see how it would go.

I generated the training data in a very similar way, except now data samples would alternate between 4 bots. The label for each sample wouldn't be the move that the bot made after that, but rather a perfect response to said move. The model would therefore be optimized to directly predict the best move to make in any position, reducing the extra steps.

You can familiarize yourself with my final datagen code by checking out the `datagen.py` file in this repository. One thing worth nothing is that 50% of time, it will generate a sample against Abbey, and only in the remaining cases it will choose 1 of 4 bots at random. So really Abbey makes up about 62.5% of the training dataset generated this way. Since the other bots are much easier to beat, it stands to reason that the model will not require quite so much data on them.

For good measure, the dataset I generated was also almost 0.5 gigabytes in size, and I trained the model for 100 epochs.

Towards the end, it demonstrated prediction accuracy of ~90% during validation, which was considerable more than I had anticipated. My immediate thought, however, was that this simply comes from high success rate against 3 simple bots.

Immediately I paired it against Abbey, since that was the critical matchup that everything else hinged on. And to my profound shock, it demonstrated a winrate of over 90%! I ran it over an over again, and the winrate did fluctuate quite a lot. However, throughout all my tests it never dropped below 70%, and on average it was ~85%.

I double-checked, triple-checked, and quite frankly, quadruple-checked my code, to make sure I didn't mess anything up and what I observed was the real data. It was! I also verified that it performed well against other bots, and sure enough, against every other one it scored ~90% winrates consistently.

As such, I dubbed my new model "Obliterator", and as you may expect, it remains my final solution to this project.

You may find Obliterator's file at `model/obliterator.keras` in this repository. There is also exactly the same model in `.h5` format, which is loaded into `RPS.py` and used to make moves. I had to switch formats when deploying my solution for Replit, because for some reason it would fail to load the `.keras` model...

You may also find the exact dataset I used to train it with at `data/obliterator_training_data.zip` - this compressed archive contains appropriately named `obliterator_training_data.json`. The code I used to create and train the model is located in `pain_train.py`.

## Advanced Tests

To better express the requirement of not using the knowledge of exact order of bots and number of games against each, I have designed an additional test case. It is located in `test/advanced_tests.py`, and can be executed from `main.py` in the same way as the original test module.

In my test case, opponent bots are played in a random order, and each matchup consists of a random number of games between 256 and 1024. The test will keep selecting the bots and pairing them against the player until at least 8000 games total are played. It will keep track of player's winrate against each bot in each separate matchup. To pass the test, player's winrate must be at least 60% in every individual matchup.

## Final Remarks

Many questions remain unanswered upon completion of this project. Did the discrepancy in Antibot validation accuracy and deployment performance really come from the patterns in pseudo-random numbers? Why exactly was Obliterator able to perform so much better than the Antibot, even though it was trained to be more "generalized"? I consider my job here well beyond done, so we may never know. But these things may be subject to further research, should there be anyone out there eager enough to try it.

Well, anyways, that was fun while it lasted. Despite this assignment seemingly going out of its way to frustrate every poor soul tasked with completing it, I found an opportunity to challenge myself with it in an interesting way, and extract much value out of my journey. If you presently find yourself trying to complete it, I hope my work and this document can serve as an inspiration for your solution, or more importantly - an inspiration for you, to not give up. Such is the stubborn human nature, after all.

Of course, don't copy my code exactly - academic honesty and everything. You know the drill.

I would also like to provide credit to whoever formulated this assignment and put the Abbey bot in it. You, sir or madam, are truly doing the devil's work. I hope you are proud of yourself.

Finally, I would like to thank you for reading this far. I hope you enjoyed my write-up, and I wish you a wonderful day.

Cheers!