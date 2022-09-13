import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from random import sample
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import random
import re

NOTE_PREFIXES = {
            'NP': ['The definition of [NP] is', 'The main function of [NP] is', '[NP] is'],
            'VP': ['[VP] means', 'After [VP],', 'Before [VP],'],
            'NNP': ['[NNP] is a', 'After this, [NNP]', '[NNP] did this because', '[NNP] felt']
        }


class QAExample(object):
    def __init__(self, id=None, context="", question=None,
                 key_phrases=None, notes=None, choices=None, label=None, is_skipped=False, statement=None, ):
        self.id = id
        self.context = context
        self.question = question
        self.statement = statement
        self.key_phrases = key_phrases
        self.notes = notes
        self.choices = choices
        self.label = label
        self.is_skipped = is_skipped


class DataProcessor(object):
    def __init__(self, args):
        self.data_type = args.task_name
        self.data_dir = args.data_dir
        self.is_training = args.is_training

    def load_examples(self, data_file, return_fields=False):
        input_file = os.path.join(self.data_dir, data_file)
        examples = []
        all_fields = []
        with open(input_file) as f_in:
            for idx, line in enumerate(tqdm(f_in, desc='Loading examples')):
                fields = json.loads(line.strip())
                example = self.convert_fields_to_example(fields)
                example.id = idx
                if example.is_skipped:
                    continue
                all_fields.append(fields)
                examples.append(example)
        if return_fields:
            return examples, all_fields
        return examples

    def convert_examples_to_features(self, examples):
        features = []
        for example in examples:
            features.append(self.convert_example_to_feature(example))
        return features

    def convert_fields_to_example(self, fields):
        pass

    def convert_example_to_feature(self, example):
        pass


class SocialIQAProcessor(DataProcessor):
    def __init__(self, args):
        super(SocialIQAProcessor, self).__init__(args)
        self.label_dict = {"A": 0, "B": 1, "C": 2}
        self.QUESTION_TO_ANSWER_PREFIX = {
            "What will (.*) want to do next?": r"As a result, [SUBJ] wanted to",
            "What will (.*) want to do after?": r"As a result, [SUBJ] wanted to",
            "How would (.*) feel afterwards?": r"As a result, [SUBJ] felt",
            "How would (.*) feel as a result?": r"As a result, [SUBJ] felt",
            "What will (.*) do next?": r"As a result, [SUBJ] will",
            "How would (.*) feel after?": r"As a result, [SUBJ] felt",
            "How would you describe (.*)\\?": r"[SUBJ] is seen as",
            "What kind of person is (.*)\\?": r"[SUBJ] is seen as",
            "How would you describe (.*) as a person?": r"[SUBJ] is seen as",
            "Why did (.*) do that?": r"Because [SUBJ] wanted to",
            "Why did (.*) do this?": r"Because [SUBJ] wanted to",
            "Why did (.*) want to do this?": r"Because [SUBJ] wanted to",
            "What does (.*) need to do beforehand?": r"Before, [SUBJ] needed to",
            "What does (.*) need to do before?": r"Before, [SUBJ] needed to",
            "What does (.*) need to do before this?": r"Before, [SUBJ] needed to",
            "What did (.*) need to do before this?": r"Before, [SUBJ] needed to",
            "What will happen to (.*)\\?": r"After, [SUBJ] will",
            "What will happen to (.*) next?": r"After, [SUBJ] will"
        }

        self.note_prefixes = NOTE_PREFIXES

    def convert_fields_to_example(self, fields):
        context = fields['context'].replace('%', ' percent')
        context = re.sub(" +", " ", context).strip()
        question = [fields['question']]
        choices = []
        label = fields.get("correct", None)
        label = self.label_dict.get(label, None) if label is not None else None
        key_phrases = fields.get("keyphrases", None)
        notes = fields.get("notes", None)
        statement = self.convert_question_to_statement(fields['question'])
        for i in self.label_dict.keys():
            ans_key = "answer" + i
            ans = fields.get(ans_key, "")
            choices.append(ans)
        return QAExample(context=context,
                         question=question,
                         statement=statement,
                         key_phrases=key_phrases,
                         notes=notes,
                         choices=choices,
                         label=label)

    def convert_question_to_statement(self, question):
        answer_prefix = ""
        for template, ans_prefix in self.QUESTION_TO_ANSWER_PREFIX.items():
            m = re.match(template, question)
            if m is not None:
                answer_prefix = ans_prefix.replace("[SUBJ]", m.group(1))
                break
        if answer_prefix == "":
            answer_prefix = question.replace("?", "is")
        return answer_prefix

    def convert_example_to_feature(self, example, take_notes=False):
        new_choices = [choice[0].lower() + choice[1:] + '.' for choice in example.choices]
        context = example.context
        prompts_for_choices = [context + " " + example.statement + " " + choice
                               for choice in new_choices]

        if example.statement.startswith("Before,"):
            state_prefix = example.statement.replace("Before, ", "")
            prompts_for_context = [state_prefix + " " + choice + " After, " + example.context
                                   for choice in example.choices]
        elif example.statement.startswith("After,"):
            state_prefix = example.statement.replace("After, ", "")
            prompts_for_context = [state_prefix + " " + choice + " Before, " + example.context
                                   for choice in new_choices]
        elif example.statement.startswith("As a result,"):
            state_prefix = example.statement.replace("As a result, ", "")
            prompts_for_context = [state_prefix + " " + choice + " Because " + example.context
                                   for choice in new_choices]
        elif example.statement.startswith("Because"):
            state_prefix = example.statement.replace("Because ", "")
            prompts_for_context = [state_prefix + " " + choice + " That is why " + example.context
                                   for choice in new_choices]
        else:
            prompts_for_context = [example.statement + " " + choice + " Considering that " + example.context
                                   for choice in new_choices]


        prompts_for_choices = [self.remove_redundancy(prompt) for prompt in prompts_for_choices]
        prompts_for_context = [self.remove_redundancy(prompt) for prompt in prompts_for_context]

        return InputFeature(prompts_for_choices, prompts_for_context,
                            example.context, new_choices)

    def remove_redundancy(self, sentence):
        return sentence.replace("wanted to want", "want").replace("needed to need", "need").\
            replace("wanted to need", "need").replace("needed to want", "want").replace("to to", "to")


class CopaProcessor(DataProcessor):
    def __init__(self, args):
        super(CopaProcessor, self).__init__(args)
        self.question_dict = {
            "effect": ["What happened after?", "What is the effect of this?",
                       "What is the possible result of this?", "So?"],
            "cause": ["What is the cause?", "What happened before this?",
                      "What is the possible reason of this?", "Why?"]
        }
        self.ques_to_state = {
            "effect": "As a result,",
            "cause": "Because"
        }
        self.note_prefixes = NOTE_PREFIXES

    def convert_fields_to_example(self, fields):
        context = fields['premise'].replace('%', ' percent')
        context = re.sub(" +", " ", context).strip()
        question = self.question_dict[fields['question']]
        choices = []
        label = fields.get("label", None)
        label = int(label) if label is not None else None
        key_phrases = fields.get("keyphrases", None)
        notes= fields.get("notes", None)
        statement = self.ques_to_state[fields['question']]
        for i in range(1, 3):
            ans_key = "choice" + str(i)
            ans = fields.get(ans_key, "")
            choices.append(ans)
        return QAExample(context=context,
                         question=question,
                         statement=statement,
                         key_phrases=key_phrases,
                         notes=notes,
                         choices=choices,
                         label=label)

    def convert_example_to_feature(self, example):
        context = example.context

        prompts_for_choices = [context + " " + example.statement + " " + choice[0].lower() + choice[1:]
                               for choice in example.choices]
        conjunction_reverse = {
            "As a result,": "Because ",
            "Because": "As a result, "
        }
        new_context = example.context[0].lower() + example.context[1:] if example.context[0] != 'I' else example.context
        prompts_for_context = [choice + " " + conjunction_reverse[example.statement] + new_context
                               for choice in example.choices]
        return InputFeature(prompts_for_choices, prompts_for_context,
                            new_context, example.choices)


class ROCStoryProcessor(DataProcessor):
    def __init__(self, args):
        super(ROCStoryProcessor, self).__init__(args)
        self.question = 'How the story ends?'
        self.statement = 'As a result,'
        self.note_prefixes = NOTE_PREFIXES

    def convert_fields_to_example(self, fields):
        context = fields['context'].replace('%', ' percent')
        context = re.sub(" +", " ", context).strip()
        question = [self.question]
        choices = fields['choices']
        label = fields.get("label", None)
        label = int(label) if label is not None else None
        key_phrases = fields.get("keyphrases", None)
        notes = fields.get("notes", None)
        statement = self.statement
        return QAExample(context=context,
                         question=question,
                         statement=statement,
                         key_phrases=key_phrases,
                         notes=notes,
                         choices=choices,
                         label=label)

    def convert_example_to_feature(self, example):
        context = example.context

        prompts_for_choices = [context + " " + example.statement + " " + choice[0].lower() + choice[1:]
                               for choice in example.choices]
        sentences = example.context.split('.')
        if len(sentences[-1].strip()) < 2:
            pre_sentences = '.'.join(sentences[:-2]) + '. '
            last_sentence = sentences[-2].strip() + '.'
        else:
            pre_sentences = '.'.join(sentences[:-1]) + '. '
            last_sentence = sentences[-1].strip() + '.'
        prompts_for_context = [pre_sentences + choice + ' Because ' + last_sentence
                               for choice in example.choices]
        return InputFeature(prompts_for_choices, prompts_for_context,
                            last_sentence, example.choices)

def lemma(sentence):
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    tokens = sentence.split(' ')
    try:
        tagged_sent = pos_tag(tokens)
    except:
        return None, None

    wnl = WordNetLemmatizer()
    lemmas_sent, tags = [], []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        tags.append(wordnet_pos)
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    return ' '.join(lemmas_sent), tags


def default_convert_question_to_statement(question):
    conj_words = ['if', 'during', 'because', 'despite', 'when', 'while', 'before', 'after', 'once', 'since']
    pre_question = ''
    question = question.strip()
    question = question[:-1].strip() if question[-1] == '?' else question
    question = question[0].lower() + question[1:]
    question_words = question.split(' ')
    for word in conj_words:
        if word == question_words[0]:
            if question.find(',') > 0:
                pre_question = question[: question.find(',') + 1]
                break
        elif word in question_words[2:]:
            word_index = question_words.index(word)
            pre_question = ' '.join(question_words[word_index:] + [','])
            question_words = question_words[:word_index]
            break
    question = ' '.join(question_words)
    lemma_question, tags = lemma(question)
    if lemma_question is None:
        return None

    curr_question = ' ' + ' '.join(question_words) + ' '
    curr_question = curr_question.replace(' you ', ' I ').replace('yourself', 'myself').replace(' your ', ' my ').strip()
    question_words = curr_question.split(' ')

    if question_words[0] == 'what':
        if len(lemma_question.split(' ')) <= 4 and 'happen' in lemma_question:
            # new_question = ''
            new_question = ' '.join(['it'] + question_words[1:] + ['that'])
        elif len(lemma_question.split(' ')) == 4 and ' '.join(
                lemma_question.split(' ')[:2] + [lemma_question.split(' ')[3]]) == 'what do do':
            new_question = question_words[2]
            if new_question == 'you':
                new_question = 'I'
        elif 'a possible reason' in ' '.join(question_words[:6]):
            new_question = ' '.join(question_words[question_words.index('reason') + 1:] + ['because'])
        elif tags[1] == 'n' and lemma_question.split(' ')[1] not in ['may', 'have', 'do', 'be', 'will', 'can']:
            new_question = ' '.join(['the'] + question_words[1:] + ['is', 'that'])
        else:
            new_question = ' '.join(['it'] + question_words[1:] + ['that'])
    elif question_words[0] == 'why':
        new_question = ' '.join(question_words[2:] + ['because'])
    elif question_words[0] == 'who':
        new_question = 'the person ' + ' '.join(question_words) + ' is'
    elif question_words[0] == 'how':
        if lemma_question.split(' ')[1] in ['be', 'can', 'will', 'may']:
            verb_idx = -1
            for idx, tag in enumerate(tags):
                if tag == 'v':
                    verb_idx = verb_idx
                    break
            if verb_idx > 2:
                new_question = ' '.join(question_words[2: verb_idx]) + ' ' + question_words[1] + ' ' \
                           + ' '.join(question_words[verb_idx:])
            else:
                new_question = ' '.join(question_words[2:])
        elif lemma_question.split(' ')[1] == 'do':
            new_question = ' '.join(question_words[2:])
        elif lemma_question.split(' ')[1] in ['long', 'far', 'often', 'old']:
            new_question = ' '.join(question_words[3:])
        elif lemma_question.split(' ')[1] in ['many', 'much']:
            if lemma_question.split(' ')[2] in ['be', 'do', 'can', 'will', 'may']:
                new_question = ' '.join(question_words[3:])
            else:
                new_question = ' '.join(question_words[2:])
        else:
            new_question = ' '.join(question_words) + ' is'
    elif question_words[0] == 'where':
        new_question = 'the place ' + ' '.join(question_words) + ' is'
    else:
        new_question = ' '.join(question_words) + ' ?'

    if new_question == '':
        new_question = pre_question
    elif pre_question != '':
        new_question = pre_question + ' ' + new_question
    new_question = new_question.strip()
    new_question = new_question[0].upper() + new_question[1:]
    return new_question


class QADataset(Dataset):
    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]


class InputFeature(object):
    def __init__(self, prompts_for_choices, prompts_for_context=None,
                 context=None, choices=None):
        self.prompts_for_choices = prompts_for_choices
        self.prompts_for_context = prompts_for_context
        self.context = context
        self.choices = choices

    def print(self):
        print('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))


