#! /usr/bin/env python
# -*- coding: utf-8 -*-
from . import extractor
from nltk.corpus import stopwords
stopword_dict = set(stopwords.words('english'))
# from stanfordcorenlp import StanfordCoreNLP
# en_model = StanfordCoreNLP(r'E:\Python_Files\stanford-corenlp-full-2018-02-27',quiet=True)
class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, en_model, text=""):
        """
        :param is_sectioned: If we want to section the text.
        :param en_model: the pipeline of tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS',
                                'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

        self.tokens = []
        self.tokens_tagged = []
        self.tokens = en_model.word_tokenize(text)
        self.tokens_tagged = en_model.pos_tag(text)
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stopword_dict:
                if token.lower() in ['is', 'are', 'was', 'were', 'be']:
                    self.tokens_tagged[i] = (token, "CC")
                else:
                    self.tokens_tagged[i] = (token, "IN")
        self.keyphrase_candidate = extractor.extract_candidates(self.tokens_tagged, en_model)
