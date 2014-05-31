=Poker CUDA Equity Simulator=

This is a small library that uses CUDA to simulate Texas Hold'Em poker equity. There is also a
hand evaluator that processes a hand of 7 cards and returns a numeric value which can be used to 
compare two or more hands and determine which is higher.

It supports simulating 2-10 players and returns the following statistics for each player:
* Wins
* Losses
* Tie
* Counter for each hand rank

The project compiles under Visual Studio 2012, 64bit only and requires CUDA 6.
