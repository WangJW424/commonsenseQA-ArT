#! /usr/bin/env python
# -*- coding: utf-8 -*-

import nltk

#GRAMMAR1 is the general way to extract NPs

GRAMMAR1 = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR2 = """  NP:
        {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR3 = """  NP:
        {<NN|NNS|JJ|VBG|VBN>*<NN|NNS>}  # Adjective(s)(optional) + Noun(s)"""

# following CFG rules are designed by us for commonsense QA
GRAMMAR4 = r"""  
        VP: {<VB.*><IN>{0,2}<NN|NNS|JJ.*>*<NN|NNS>} # Adverb(optional) + Adjective(s)(optional) + Noun(s)
        NP: {<NN|NNS|JJ.*>*<NN|NNS>} # Adjective(s)(optional) + Noun(s)   
        NNP: {<NNP>{1,2}}    # Person Name
        """


def extract_candidates(tokens_tagged, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """
    np_parser = nltk.RegexpParser(GRAMMAR4)  # Noun phrase parser
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label in ("NP", "VP", "NNP")):
            np = ' '.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            keyphrase_candidate.append((np, start_end, token._label))

        else:
            count += 1

    return keyphrase_candidate