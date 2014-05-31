
#ifndef COMMON
#define COMMON

typedef unsigned long long ulong;
typedef unsigned char byte;

#pragma pack(push, 4)
typedef struct
{
	int Wins;
	int Losses;
	int Draws;

	// Counts the number of hand ranks for each rank.
	// Key is the HandRank (1 = high card... 9 = straight flush)
	int Hands[10];

} SimResult;
#pragma pack(pop)

// Defines for card suit values
#define Spade 0
#define Heart 1
#define Diamond 2
#define Club 3

// Defines for card ranks.
#define _2 2
#define _3 3
#define _4 4
#define _5 5
#define _6 6
#define _7 7
#define _8 8
#define _9 9
#define _10 10
#define Jack 11
#define Queen 12
#define King 13
#define Ace 14

// Defines for hand rank.
#define HighCard 1
#define Pair 2
#define TwoPair 3
#define ThreeOfaKind 4
#define Straight 5
#define Flush 6
#define FullHouse 7
#define FourOfaKind 8
#define StraightFlush 9

// Defines for packing and extracting card data into a byte.
#define GetTypeFromValue(value) (value >> 20)
#define Rank(value) (value >> 2)
#define Suit(value) (value & 0x03)
#define Pack(rank, suit) ((rank << 2) | suit)

// Size of the random data array. Note that this is the number of 64-bit ulong values,
// so the byte count is 8x larger.
#define DATA_SIZE 10000000

// Setting one or both cards of a player to this value will instruct the simulator 
// to treat it as an empty seat.
#define EMPTY_SEAT 255

// CUDA Settings.
#define MAX_CUDA_THREADS 1024
#define MAX_CUDA_BLOCKS 12
#define MAX_CUDA_ITERATIONS 384
#define CUDA_PARALLEL_COUNT (MAX_CUDA_THREADS * MAX_CUDA_BLOCKS)

// Native Settings. 
#define MAX_NATIVE_THREADS 16

#ifdef DLLEXPORT
#define DLLDIRECTION __declspec(dllexport)
#else
#define DLLDIRECTION __declspec(dllimport)
#endif

namespace HandEval
{
	extern "C"
	{
		// Evaluate returns an integer score for a hand. 
		// The score can be used to compare two hands of seven cards each, and the higher value is the winner.
		// Two equal values represent a draw.
		// You can use GetTypeFromValue() to extract the hand rank.
		//
		// cards must be an array of exactly 7 cards
		DLLDIRECTION __host__ __device__ int Evaluate(unsigned const char* cards);
	}
}

namespace TableSimCuda
{
	extern "C"
	{
		// This function must be called before any calls to SimulateCuda are made.
		// The argument must be a pointer to an array of ulong with length DATA_SIZE
		DLLDIRECTION void InitCuda(const ulong* inputData);

		// Invoke simulation.
		// boardData: length-25 array. First 5 bytes are community cards, followed by 2 cards for each player.
		//		A seat can be marked as empty using the SEAT_EMPTY value for both of the players cards
		// simCount: Number of times to run the simulation.
		// results: Length-10 array of SimResult structs that get filled with statistics
		DLLDIRECTION int SimulateCuda(const byte* boardData, int simCount, SimResult* results);
	}
}

namespace TableSimNative
{
	extern "C"
	{
		// This function must be called before any calls to Simulate are made.
		// The argument must be a pointer to an array of ulong with length DATA_SIZE
		DLLDIRECTION void Init(const ulong* inputData);

		// Invoke simulation.
		// boardData: length-25 array. First 5 bytes are community cards, followed by 2 cards for each player.
		//		A seat can be marked as empty using the SEAT_EMPTY value for both of the players cards
		// simCount: Number of times to run the simulation.
		// thread: The function is thread safe, but to ensure each thread uses a unique random number generator, 
		//		pass in a different number for each thread, between 0...MAX_NATIVE_THREADS
		// results: Length-10 array of SimResult structs that get filled with statistics
		DLLDIRECTION int Simulate(const byte* boardData, int simCount, int thread, SimResult* results);

		// Get a shuffled deck of 52 cards
		// thread: The function is thread safe, but to ensure each thread uses a unique random number generator, 
		//		pass in a different number for each thread, between 0...MAX_NATIVE_THREADS
		// output: length-52 array that is to be filled with shuffled cards
		DLLDIRECTION void GetDeck(int thread, byte* output);

		// Returns a random byte in range [0...max[
		// thread: The function is thread safe, but to ensure each thread uses a unique random number generator, 
		//		pass in a different number for each thread, between 0...MAX_NATIVE_THREADS
		DLLDIRECTION byte GetRandom(int thread, int max);
	}
}

#endif
