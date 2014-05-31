
#include "HandEval.h"
#include <string.h>

namespace TableSimCuda
{
	using namespace HandEval;

	__device__ byte GetRandom(int thread, int max);
	__device__ void GetDeck(int thread, byte* output);
	__device__ bool LoadHand(byte* board, int i, byte* hand);
	__device__ void DealBoard(byte* board, int thread);

	const int BoardBytes = 25; // maximum size of the board, with 10 players
	const ulong MaxSampleSize = 0xFFFFFFFFFFFFFFFF >> 2;

	__device__ byte cCardValues[52];
	__device__ ulong cRandomData[DATA_SIZE];
	__device__ ulong cSamples[CUDA_PARALLEL_COUNT];
	__device__ int cIndex[CUDA_PARALLEL_COUNT];

	void InitCuda(const ulong* inputData)
	{
		cudaError_t cudaStatus;
		byte cardValues[52];
		ulong samples[CUDA_PARALLEL_COUNT];
		int index[CUDA_PARALLEL_COUNT];

		cudaDeviceProp properties;
		cudaStatus = cudaSetDevice(0);
		cudaGetDeviceProperties(&properties, 0);

		for (int i = 0; i < 52; i++)
		{
			int rank = (i % 13) + 2;
			int suit = i / 13;
			cardValues[i] = (byte)((rank << 2) | suit);
		}

		for (int i = 0; i < CUDA_PARALLEL_COUNT; i++)
		{
			int idx = (int)((ulong)DATA_SIZE * (ulong)i / (ulong)CUDA_PARALLEL_COUNT);
			samples[i] = inputData[idx];
			index[i] = (idx + 1) % DATA_SIZE;
		}
	
		cudaStatus = cudaMemcpyToSymbol(cCardValues, cardValues, 52);
		cudaStatus = cudaMemcpyToSymbol(cRandomData, inputData, DATA_SIZE * sizeof(ulong));
		cudaStatus = cudaMemcpyToSymbol(cSamples, samples, CUDA_PARALLEL_COUNT * sizeof(ulong));
		cudaStatus = cudaMemcpyToSymbol(cIndex, index, CUDA_PARALLEL_COUNT * sizeof(int));
	}

	// pid = cude parallel id (block * blocksize + theadId)
	byte GetRandom(int pid, int max)
	{
		int idx = cIndex[pid];
		ulong sample = (cSamples[pid] + cRandomData[idx]) % MaxSampleSize;
		idx++;
		if (idx >= DATA_SIZE) idx -= DATA_SIZE;
		cIndex[pid] = idx;
		cSamples[pid] = sample;

		byte card = (byte)(sample % (ulong)max);
		return card;
	}

	// pid = cude parallel id (block * blocksize + theadId)
	void GetDeck(int pid, byte* output)
	{
		output[0] = (byte)((2 << 2) + (0));

		for (int i = 1; i < 52; i++)
		{
			byte j = GetRandom(pid, i);
			output[i] = output[j];
			output[j] = cCardValues[i];
		}
	}

	__global__ void SimKernel(const byte* boardData, const int simCount, SimResult* resultsArray)
	{
		byte board[BoardBytes];

		int thread = threadIdx.x;
		int pid = blockIdx.x * blockDim.x + thread;
		SimResult* result = &resultsArray[10 * pid];

		for (int run = 0; run < simCount; run++)
		{
			for (int i = 0; i < BoardBytes; i++)
				board[i] = boardData[i];

			DealBoard(board, pid);

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

	int SimulateCuda(const byte* boardData, int simCount, SimResult* results)
	{
		int blocks = MAX_CUDA_BLOCKS;
		int threads = MAX_CUDA_THREADS;
		int iterations = simCount / blocks / threads;
		int actualSimCount = blocks * threads * iterations;
		int iterationsRemaining = iterations;

		cudaError_t cudaStatus;
		byte* cBoardData;
		cudaStatus = cudaMalloc((void**)&cBoardData, BoardBytes);
		cudaStatus = cudaMemcpy(cBoardData, boardData, BoardBytes, cudaMemcpyHostToDevice);

		SimResult* cResults;
		cudaStatus = cudaMallocManaged(&cResults, 10 * CUDA_PARALLEL_COUNT * sizeof(SimResult));
		cudaMemset(cResults, 0, 10 * CUDA_PARALLEL_COUNT * sizeof(SimResult));

		while(iterationsRemaining > 0)
		{
			if (iterationsRemaining > MAX_CUDA_ITERATIONS)
			{
				SimKernel<<<blocks, threads>>>(cBoardData, MAX_CUDA_ITERATIONS, cResults);
				iterationsRemaining -= MAX_CUDA_ITERATIONS;
			}
			else
			{
				SimKernel<<<blocks, threads>>>(cBoardData, iterationsRemaining, cResults);
				iterationsRemaining = 0;
			}

			cudaStatus = cudaGetLastError();
			cudaStatus = cudaDeviceSynchronize();
		}
	
		memset(results, 0, 10 * sizeof(SimResult));

		// add up results
		for (int i = 0; i < blocks * threads; i++)
		{
			for (int p = 0; p < 10; p++)
			{
				SimResult* res = &cResults[10 * i + p];

				results[p].Wins += res->Wins;
				results[p].Losses += res->Losses;
				results[p].Draws += res->Draws;
				results[p].Hands[0] += res->Hands[0];
				results[p].Hands[1] += res->Hands[1];
				results[p].Hands[2] += res->Hands[2];
				results[p].Hands[3] += res->Hands[3];
				results[p].Hands[4] += res->Hands[4];
				results[p].Hands[5] += res->Hands[5];
				results[p].Hands[6] += res->Hands[6];
				results[p].Hands[7] += res->Hands[7];
				results[p].Hands[8] += res->Hands[8];
				results[p].Hands[9] += res->Hands[9];
			}
		}

		cudaStatus = cudaFree(cBoardData);
		cudaStatus = cudaFree(cResults);
		return actualSimCount;
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

	/// Deals cards in replacement for any wildcards on the table
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
