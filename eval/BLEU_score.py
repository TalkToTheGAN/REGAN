#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
from functools import *
from operator import mul
from collections import Counter
from fractions import Fraction


isDebug = False

#### Helper Functions #####
def get_ngrams(words, n):
    history = []
    seq = iter(words)
    while n > 1:
        history.append(next(seq))
        n -= 1

    for word in words:
        history.append(word)
        yield tuple(history)
        del history[0]

def get_bigrams(words):
    for item in get_ngrams(words, 2):
        yield item

def get_trigrams(words):
    for item in get_ngrams(words, 3):
        yield item

def __get_closest_ref_length(references, candidate_len):

    ref_lens = (len(ref) for ref in references)
    closest = min(ref_lens, key=lambda  ref_len:(abs(ref_len - candidate_len), ref_len))

    return closest

def __get_BP(closest_ref_len, candidate_len):

    if candidate_len == 0:
        return 0

    brevity = closest_ref_len / candidate_len
    return 1 if (candidate_len > closest_ref_len) else math.exp(1 - brevity)

def __get_precision(candidate, references, n):
    '''
    Note: Modified precision with clip/cap value set to 2

    :param candidate: candidate sentence
    :param references:
    :param n: the n gram order, int
    :return: Modified precision for the nth order ngram
    '''
    words_c = candidate.split()
    lc = len(words_c)
    counts = Counter(get_ngrams(words_c, n)) if lc >= n else Counter()

    max_counts = {}
    for reference in references:
        words_ref = reference.split()
        reference_counts = Counter(get_ngrams(words_ref, n)) if len(words_ref) >= n else Counter()
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    reference_counts[ngram])

    # Get Intersection between candidate and references' counts.
    clipped_counts = {ngram: min(count, max_counts[ngram])
                      for ngram, count in counts.items()}

    numerator = sum(clipped_counts.values())
    # Avoid ZeroDivisionError. Usually when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator)


def BLEU_score(candidate, references, n):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.

	
	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""
    hasMultipleReferences = True if len(references) > 1 else False
    if isDebug:
        print(f"\n##### BLEU_score(candidate, references, n = {n} ##### ")
        print('-' * 25)
        print(f"Candidate sentence:   \t{candidate}")
        print(f"Handsard ref sentence:\t{references[0]}")
        if hasMultipleReferences: print(f"Google ref sentence:  \t{references[1]}")
        print('-' * 25)

    bleu_score = 0.0
    n_precisions = []

    ref_1_sent = references[0]
    ref_1_words = ref_1_sent.split()
    if hasMultipleReferences:
        ref_2_sent = references[1]
        ref_2_words = ref_2_sent.split()

    candidate_words = candidate.split()

    ## Compute N-Gram Precision, i.e. p1, p2, p3
    for i in range(1,n+1):
        p_i = __get_precision(candidate, references, n=i)
        n_precisions.append(float(p_i))

    p_score = reduce(mul, n_precisions) ** (1/n)
    if isDebug: print(f"p_score = {p_score}")

    ## Compute BP (Brevity Penalty)
    lc = len(candidate_words)
    l_r1 = len(ref_1_words)
    l_r2 = len(ref_2_words) if hasMultipleReferences else 0
    dist_r1 = abs(l_r1 - lc); dist_r2 = abs(l_r2 - lc)

    ri = l_r1 if dist_r1 <= dist_r2 else l_r2

    brevity = ri / lc
    if isDebug: print(f"l_r1 = {l_r1} \t l_r2 = {l_r2} \t lc = {lc}\tbrevity = {brevity}")

    BP = __get_BP(ri, lc)
    if isDebug: print(f"BP = {BP}")

    bleu_score = round((BP * p_score), 4)

    return bleu_score