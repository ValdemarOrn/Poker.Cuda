
#include "FastEval.h"

const int RES_LEN = 6;
const byte NOVAL = 255;

__host__ __device__ void Eval(unsigned const char* cards, byte* result);
__host__ __device__ byte getStraight(unsigned const char* rankCount);
__host__ __device__ byte countSuit(unsigned const char* cards, byte* suitOutput);
__host__ __device__ void countRank(unsigned const char* cards, byte* rankOutput, int filterBysuit);
__host__ __device__ void sort(const byte* rankCount, byte* output, byte* handRankFlags);

__host__ __device__ void EvalHighCard(byte* ranks, byte* output);
__host__ __device__ void EvalOnePair(byte* ranks, byte* output);
__host__ __device__ void EvalTwoPair(byte* ranks, byte* output);
__host__ __device__ void EvalThreeOfaKind(byte* ranks, byte* output);
__host__ __device__ void EvalFullHouse(byte* ranks, byte* output);
__host__ __device__ void EvalFourOfaKind(byte* ranks, byte* output);

__host__ __device__ int Evaluate(unsigned const char* cards)
{
	byte result[RES_LEN] = { 0 };
	Eval(cards, result);

	int value = (result[0] << 20) | (result[1] << 16) | (result[2] << 12) | (result[3] << 8) | (result[4] << 4) | (result[5] << 0);
	return value;
}

void Eval(unsigned const char* cards, byte* result)
{
	byte tempResult[RES_LEN] = { 0 };

	byte sortedRanks[7] = { 0 }; // sorted cards
	byte handRankFlags[5] = { 0 }; // counts number of pairs, sets and quads
	byte suitCount[4] = { 0 }; // count cards of each suit
	byte rankCount[15] = { 0 }; // count cards of each rank
	byte rankCountSuited[15] = { 0 }; // count cards of each rank in suit

	byte flushSuit = NOVAL; // suit of flush, NOVAL if no flush
	byte hiStraight = NOVAL; // high card of straight, NOVAL if no straight
	byte hiStraightFlush = NOVAL; // high card of straight flush, NOVAL if no straight

	countRank(cards, rankCount, NOVAL);
	sort(rankCount, sortedRanks, handRankFlags);
	flushSuit = countSuit(cards, suitCount);
	hiStraight = getStraight(rankCount);

	if (flushSuit != NOVAL)
		countRank(cards, rankCountSuited, flushSuit);

	if (hiStraight != NOVAL && flushSuit != NOVAL)
		hiStraightFlush = getStraight(rankCountSuited);

	// find result
	if (hiStraightFlush != NOVAL)
	{
		tempResult[0] = StraightFlush;
		tempResult[1] = hiStraightFlush;
	}
	else if (handRankFlags[4] > 0)
	{
		EvalFourOfaKind(sortedRanks, tempResult);
	}
	else if ((handRankFlags[3] == 1 && handRankFlags[2] > 0) || handRankFlags[3] == 2)
	{
		EvalFullHouse(sortedRanks, tempResult);
	}
	else if (flushSuit != NOVAL)
	{
		sort(rankCountSuited, sortedRanks, handRankFlags);
		EvalHighCard(sortedRanks, tempResult);
		tempResult[0] = Flush;
	}
	else if (hiStraight != NOVAL)
	{
		tempResult[0] = Straight;
		tempResult[1] = hiStraight;
	}
	else if (handRankFlags[3] > 0)
	{
		EvalThreeOfaKind(sortedRanks, tempResult);
	}
	else if (handRankFlags[2] > 1)
	{
		EvalTwoPair(sortedRanks, tempResult);
	}
	else if (handRankFlags[2] > 0)
	{
		EvalOnePair(sortedRanks, tempResult);
	}
	else
	{
		EvalHighCard(sortedRanks, tempResult);
	}

	result[0] = tempResult[0];
	result[1] = tempResult[1];
	result[2] = tempResult[2];
	result[3] = tempResult[3];
	result[4] = tempResult[4];
	result[5] = tempResult[5];
}

void sort(const byte* rankCount, byte* output, byte* handRankFlags)
{
	int sortedCardCount = 0;

	for (int i = 14; i >= 2; i--)
	{
		int count = rankCount[i];

		if (count == 0)
		{
			continue;
		}
		if (count == 1)
		{
			output[sortedCardCount++] = i;
			handRankFlags[1]++;
		}
		else if (count == 2)
		{
			output[sortedCardCount++] = i;
			output[sortedCardCount++] = i;
			handRankFlags[2]++;
		}
		else if (count == 3)
		{
			output[sortedCardCount++] = i;
			output[sortedCardCount++] = i;
			output[sortedCardCount++] = i;
			handRankFlags[3]++;
		}
		else if (count == 4)
		{
			output[sortedCardCount++] = i;
			output[sortedCardCount++] = i;
			output[sortedCardCount++] = i;
			output[sortedCardCount++] = i;
			handRankFlags[4]++;
		}
	}
}

// returns the value of the highest card in a straight.
// returns NOVAL if no straight was found
byte getStraight(unsigned const char* rankCount)
{
	int counter = 0;
	byte maxCard;
	for (int i = Ace; i >= 0; i--)
	{
		if (counter == 0)
			maxCard = i;

		if (rankCount[i] > 0)
			counter++;
		else
			counter = 0;

		if (counter >= 5)
			return maxCard;
	}

	return NOVAL;
}

// fills an size-4 output with the number of cards for each suit
byte countSuit(unsigned const char* cards, byte* suitOutput)
{
	for (int i = 0; i < 7; i++)
	{
		int suit = Suit(cards[i]);
		suitOutput[suit]++;
	}

	if (suitOutput[Spade] >= 5)
		return Spade;
	if (suitOutput[Heart] >= 5)
		return Heart;
	if (suitOutput[Diamond] >= 5)
		return Diamond;
	if (suitOutput[Club] >= 5)
		return Club;

	return NOVAL;
}

// fills an size-15 (1-14) output with the number of cards for each rank
// note that output[1] == output[14] (Ace)
// If filterBysuit != -1, then it only counts the selected suit
void countRank(unsigned const char* cards, byte* rankOutput, int filterBysuit)
{
	if (filterBysuit != NOVAL)
	{
		for (int i = 0; i < 7; i++)
		{
			int rank = Rank(cards[i]);
			int suit = Suit(cards[i]);
			if (suit != filterBysuit)
				continue;

			rankOutput[rank]++;
		}
	}
	else
	{
		for (int i = 0; i < 7; i++)
		{
			int rank = Rank(cards[i]);
			rankOutput[rank]++;
		}
	}

	rankOutput[1] = rankOutput[Ace];
}


// ----------------------------------------------------------------------------------------------

void EvalHighCard(byte* ranks, byte* output)
{
	output[0] = HighCard;
	output[1] = ranks[0];
	output[2] = ranks[1];
	output[3] = ranks[2];
	output[4] = ranks[3];
	output[5] = ranks[4];
}

void EvalOnePair(byte* ranks, byte* output)
{
	if (ranks[0] == ranks[1])
	{
		output[0] = Pair;
		output[1] = ranks[0];
		output[2] = ranks[2];
		output[3] = ranks[3];
		output[4] = ranks[4];
	}
	else if (ranks[1] == ranks[2])
	{
		output[0] = Pair;
		output[1] = ranks[1];
		output[2] = ranks[0];
		output[3] = ranks[3];
		output[4] = ranks[4];
	}
	else if (ranks[2] == ranks[3])
	{
		output[0] = Pair;
		output[1] = ranks[2];
		output[2] = ranks[0];
		output[3] = ranks[1];
		output[4] = ranks[4];
	}
	else if (ranks[3] == ranks[4])
	{
		output[0] = Pair;
		output[1] = ranks[3];
		output[2] = ranks[0];
		output[3] = ranks[1];
		output[4] = ranks[2];
	}
	else if (ranks[4] == ranks[5])
	{
		output[0] = Pair;
		output[1] = ranks[4];
		output[2] = ranks[0];
		output[3] = ranks[1];
		output[4] = ranks[2];
	}
	else if (ranks[5] == ranks[6])
	{
		output[0] = Pair;
		output[1] = ranks[5];
		output[2] = ranks[0];
		output[3] = ranks[1];
		output[4] = ranks[2];
	}
}

void EvalTwoPair(byte* ranks, byte* output)
{
	if (ranks[0] == ranks[1] && ranks[2] == ranks[3])
	{
		output[0] = TwoPair;
		output[1] = ranks[0];
		output[2] = ranks[2];
		output[3] = ranks[4];
	}
	else if (ranks[0] == ranks[1] && ranks[3] == ranks[4])
	{
		output[0] = TwoPair;
		output[1] = ranks[0];
		output[2] = ranks[3];
		output[3] = ranks[2];
	}
	else if (ranks[0] == ranks[1] && ranks[4] == ranks[5])
	{
		output[0] = TwoPair;
		output[1] = ranks[0];
		output[2] = ranks[4];
		output[3] = ranks[2];
	}
	else if (ranks[0] == ranks[1] && ranks[5] == ranks[6])
	{
		output[0] = TwoPair;
		output[1] = ranks[0];
		output[2] = ranks[5];
		output[3] = ranks[2];
	}


	else if (ranks[1] == ranks[2] && ranks[3] == ranks[4])
	{
		output[0] = TwoPair;
		output[1] = ranks[1];
		output[2] = ranks[3];
		output[3] = ranks[0];
	}
	else if (ranks[1] == ranks[2] && ranks[4] == ranks[5])
	{
		output[0] = TwoPair;
		output[1] = ranks[1];
		output[2] = ranks[4];
		output[3] = ranks[0];
	}
	else if (ranks[1] == ranks[2] && ranks[5] == ranks[6])
	{
		output[0] = TwoPair;
		output[1] = ranks[1];
		output[2] = ranks[5];
		output[3] = ranks[0];
	}


	else if (ranks[2] == ranks[3] && ranks[4] == ranks[5])
	{
		output[0] = TwoPair;
		output[1] = ranks[2];
		output[2] = ranks[4];
		output[3] = ranks[0];
	}
	else if (ranks[2] == ranks[3] && ranks[5] == ranks[6])
	{
		output[0] = TwoPair;
		output[1] = ranks[2];
		output[2] = ranks[5];
		output[3] = ranks[0];
	}


	else if (ranks[3] == ranks[4] && ranks[5] == ranks[6])
	{
		output[0] = TwoPair;
		output[1] = ranks[3];
		output[2] = ranks[5];
		output[3] = ranks[0];
	}
}

void EvalThreeOfaKind(byte* ranks, byte* output)
{
	if (ranks[0] == ranks[2])
	{
		output[0] = ThreeOfaKind;
		output[1] = ranks[0];
		output[2] = ranks[3];
		output[3] = ranks[4];
	}
	else if (ranks[1] == ranks[3])
	{
		output[0] = ThreeOfaKind;
		output[1] = ranks[1];
		output[2] = ranks[0];
		output[3] = ranks[4];
	}
	else if (ranks[2] == ranks[4])
	{
		output[0] = ThreeOfaKind;
		output[1] = ranks[2];
		output[2] = ranks[0];
		output[3] = ranks[1];
	}
	else if (ranks[3] == ranks[5])
	{
		output[0] = ThreeOfaKind;
		output[1] = ranks[3];
		output[2] = ranks[0];
		output[3] = ranks[1];
	}
	else if (ranks[4] == ranks[6])
	{
		output[0] = ThreeOfaKind;
		output[1] = ranks[4];
		output[2] = ranks[0];
		output[3] = ranks[1];
	}
}

void EvalFullHouse(byte* ranks, byte* output)
{
	// high triple, low pair

	if (ranks[0] == ranks[2] && ranks[3] == ranks[4])
	{
		output[0] = FullHouse;
		output[1] = ranks[0];
		output[2] = ranks[3];
	}
	else if (ranks[0] == ranks[2] && ranks[4] == ranks[5])
	{
		output[0] = FullHouse;
		output[1] = ranks[0];
		output[2] = ranks[4];
	}
	else if (ranks[0] == ranks[2] && ranks[5] == ranks[6])
	{
		output[0] = FullHouse;
		output[1] = ranks[0];
		output[2] = ranks[5];
	}
	else if (ranks[1] == ranks[3] && ranks[4] == ranks[5])
	{
		output[0] = FullHouse;
		output[1] = ranks[1];
		output[2] = ranks[4];
	}
	else if (ranks[1] == ranks[3] && ranks[5] == ranks[6])
	{
		output[0] = FullHouse;
		output[1] = ranks[1];
		output[2] = ranks[5];
	}
	else if (ranks[2] == ranks[4] && ranks[5] == ranks[6])
	{
		output[0] = FullHouse;
		output[1] = ranks[2];
		output[2] = ranks[5];
	}


	// high pair, low triple

	else if (ranks[0] == ranks[1] && ranks[2] == ranks[4])
	{
		output[0] = FullHouse;
		output[1] = ranks[2];
		output[2] = ranks[0];
	}
	else if (ranks[0] == ranks[1] && ranks[3] == ranks[5])
	{
		output[0] = FullHouse;
		output[1] = ranks[3];
		output[2] = ranks[0];
	}
	else if (ranks[0] == ranks[1] && ranks[4] == ranks[6])
	{
		output[0] = FullHouse;
		output[1] = ranks[4];
		output[2] = ranks[0];
	}
	else if (ranks[1] == ranks[2] && ranks[3] == ranks[5])
	{
		output[0] = FullHouse;
		output[1] = ranks[3];
		output[2] = ranks[1];
	}
	else if (ranks[1] == ranks[2] && ranks[4] == ranks[6])
	{
		output[0] = FullHouse;
		output[1] = ranks[4];
		output[2] = ranks[1];
	}
	else if (ranks[2] == ranks[3] && ranks[4] == ranks[6])
	{
		output[0] = FullHouse;
		output[1] = ranks[4];
		output[2] = ranks[2];
	}
}

void EvalFourOfaKind(byte* ranks, byte* output)
{
	if (ranks[0] == ranks[3])
	{
		output[0] = FourOfaKind;
		output[1] = ranks[0];
		output[2] = ranks[4];
	}
	else if (ranks[1] == ranks[4])
	{
		output[0] = FourOfaKind;
		output[1] = ranks[1];
		output[2] = ranks[0];
	}
	else if (ranks[2] == ranks[5])
	{
		output[0] = FourOfaKind;
		output[1] = ranks[2];
		output[2] = ranks[0];
	}
	else if (ranks[3] == ranks[6])
	{
		output[0] = FourOfaKind;
		output[1] = ranks[3];
		output[2] = ranks[0];
	}
}

