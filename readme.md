# Poker CUDA Equity Simulator

This is a small library that uses CUDA to simulate Texas Hold'Em poker equity. There is also a
hand evaluator that processes a hand of 7 cards and returns a numeric value which can be used to 
compare two or more hands and determine which is higher.

It supports simulating 2-10 players and returns the following statistics for each player:
* Wins
* Losses
* Tie
* Counter for each hand rank

There is also a version of the Simulator that runs on the CPY and does not use CUDA. This function is not
multithreaded but it thread safe and can be called in parallel from multiple threads.

The project compiles under Visual Studio 2012, 64bit only and requires CUDA 6.

## Test Results

System: Intel i7 Extreme 4930K, 32GB DDR3 2400Mhz, EVGA GeForce GTX 780 Superclocked

Time for 10 Million simulations:

* Native C++ Code, running 12 threads in parallel: 7.58 sec
* Cuda Code: 0.50 sec

