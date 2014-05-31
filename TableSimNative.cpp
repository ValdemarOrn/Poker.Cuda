
#include "HandEval.h"
#include <string.h>

namespace TableSimNative
{
	using namespace HandEval;

	byte GetRandom(int thread, int max);
	void GetDeck(int thread, byte* output);
	bool LoadHand(byte* board, int i, byte* hand);
	void DealBoard(byte* board, int thread);

	const int BoardBytes = 25;
	const ulong MaxSampleSize = 0xFFFFFFFFFFFFFFFF >> 2;

	byte cardValues[52];
	ulong randomData[DATA_SIZE];
	ulong samples[MAX_NATIVE_THREADS];
	int index[MAX_NATIVE_THREADS];

	void Init(const ulong* inputData)
	{
		for (int i = 0; i < 52; i++)
		{
			int rank = (i % 13) + 2;
			int suit = i / 13;
			cardValues[i] = (byte)((rank << 2) | suit);
		}

		for (int i = 0; i < MAX_NATIVE_THREADS; i++)
		{
			int idx = (int)((ulong)DATA_SIZE * (ulong)i / (ulong)MAX_NATIVE_THREADS);
			samples[i] = inputData[idx];
			index[i] = (idx + 1) % DATA_SIZE;
		}
	
		memcpy_s(randomData, DATA_SIZE * sizeof(ulong), inputData, DATA_SIZE * sizeof(ulong));
	}

	byte GetRandom(int thread, int max)
	{
		int idx = index[thread];
		ulong sample = (samples[thread] + randomData[idx]) % MaxSampleSize;
		idx++;
		if (idx >= DATA_SIZE) idx -= DATA_SIZE;
		index[thread] = idx;
		samples[thread] = sample;

		byte card = (byte)(sample % (ulong)max);
		return card;
	}

	/// <summary>
	/// Deals a random deck
	/// </summary>
	/// <param name="thread">a value from 0...PARALLEL_COUNT</param>
	/// <param name="output">pointer to a size 52 array</param>
	void GetDeck(int thread, byte* output)
	{
		output[0] = (byte)((2 << 2) + (0));

		for (int i = 1; i < 52; i++)
		{
			byte j = GetRandom(thread, i);
			output[i] = output[j];
			output[j] = cardValues[i];
		}
	}

	void Sim(const byte* boardData, const int simCount, const int thread, SimResult* result)
	{
		byte board[BoardBytes];

		for (int run = 0; run < simCount; run++)
		{
			for (int i = 0; i < BoardBytes; i++)
				board[i] = boardData[i];

			DealBoard(board, thread);

			int max = 0;
			int playerScore[10];
			byte hand[7];

			// eval
			for (int i = 0; i < 10; i++)
			{
				int ok = LoadHand(board, i, hand);
				if (!ok) continue;
				int rank = Evaluate(hand);
				playerScore[i] = rank;
				if (rank > max)
					max = rank;
			}

			// check for tie
			bool maxFound = false;
			bool isTie = false;
			for (int i = 0; i < 10; i++)
			{
				if (playerScore[i] == max)
				{
					if (maxFound)
					{
						isTie = true;
						break;
					}
					maxFound = true;
				}
			}

			// add up the results
			for (int i = 0; i < 10; i++)
			{
				int score = playerScore[i];
				byte type = GetTypeFromValue(score);
				if (score == max)
				{
					if (isTie)
						result[i].Draws++;
					else
						result[i].Wins++;
				}
				else
					result[i].Losses++;

				result[i].Hands[type]++;
			}
		}
	}


	/// <summary>
	/// Simulate a position n times
	/// Format of data is [5 common cards][2 cards per player * 10]
	/// Length of data must be 15
	/// </summary>
	/// <param name="data">array describing board state.</param>
	/// <param name="simCount"></param>
	/// <param name="results">array of 10 SimResult structs</param>
	int Simulate(const byte* boardData, int simCount, int thread, SimResult* results)
	{
		thread = thread % MAX_NATIVE_THREADS;
		memset(results, 0, 10 * sizeof(SimResult));
		Sim(boardData, simCount, thread, results);
		return simCount;
	}

	/// <summary>
	/// Loads the cards into an array. Returns true if hand is valid.
	/// </summary>
	/// <param name="board"></param>
	/// <param name="i"></param>
	/// <param name="hand"></param>
	/// <returns></returns>
	bool LoadHand(byte* board, int i, byte* hand)
	{
		hand[0] = board[0];
		hand[1] = board[1];
		hand[2] = board[2];
		hand[3] = board[3];
		hand[4] = board[4];
		hand[5] = board[5 + i * 2];
		hand[6] = board[6 + i * 2];

		return !(hand[0] == EMPTY_SEAT
			|| hand[1] == EMPTY_SEAT
			|| hand[2] == EMPTY_SEAT
			|| hand[3] == EMPTY_SEAT
			|| hand[4] == EMPTY_SEAT
			|| hand[5] == EMPTY_SEAT
			|| hand[6] == EMPTY_SEAT);
	}

	/// <summary>
	/// Deals for any wildcards on the table
	/// </summary>
	/// <param name="board"></param>
	void DealBoard(byte* board, int thread)
	{
		byte set[64] = { 0 };

		int iDeck = 0;
		byte deck[52];
		GetDeck(thread, deck);

		// mark used cards
		for (int i = 0; i < BoardBytes; i++)
		{
			byte card = board[i];
			if (card < 64)
				set[card] = 1;
		}

		// deal and replace wildcards
		for (int i = 0; i < BoardBytes; i++)
		{
			byte card = board[i];
			if (card == 0)
			{
				byte replacement;

				while (true)
				{
					replacement = deck[iDeck++];
					if (set[replacement] == 1)
						continue;

					break;
				}

				set[replacement] = 1;
				board[i] = replacement;
			}
		}
	}
}
