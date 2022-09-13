from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank
from stanfordcorenlp import StanfordCoreNLP
import argparse
from processors import *
import logging

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

PROCESSOR_CLASSES = {
    'copa': CopaProcessor,
    'socialiqa': SocialIQAProcessor,
    'rocstory': ROCStoryProcessor
}


def extract(args, examples, fields, extractor, SIF):
    for idx, example in enumerate(tqdm(examples, 'Extract keyphrases')):
        context = example.context
        choices = example.choices
        context_kps = SIFRank(context, SIF, extractor, N=args.max_keyphrase_num,
                             elmo_layers_weight=args.elmo_layers_weight, if_DS=True)
        choices_kps = []
        if args.extract_choices:
            for choice in choices:
                choice_kps = SIFRank(choice, SIF, extractor, N=1,
                                     elmo_layers_weight=args.elmo_layers_weight, if_DS=True)
                if len(choice_kps) == 0:
                    phrase = choice.replace('.', '')
                    choices_kps.append((phrase, (0, 0), 'ST'))
                else:
                    phrase, feature = choice_kps[0]
                    choices_kps.append((phrase, (0, 0), feature[-1]))
        all_phrase_and_pos = []
        for phrase, feature in context_kps:
            p_start = context.lower().find(phrase)
            if p_start < 0:
                continue
            p_end = p_start + len(phrase)
            if feature[-1] == 'NNP':
                phrase = phrase[0].upper() + phrase[1:]
            all_phrase_and_pos.append((phrase, (p_start, p_end), feature[-1]))
        all_phrase_and_pos += choices_kps
        fields[idx]['keyphrases'] = all_phrase_and_pos
    output_file = os.path.join(args.data_dir, args.output_file)
    with open(output_file, 'w') as f_out:
        for field in fields:
            f_out.write(json.dumps(field) + '\n')
            f_out.flush()
    logger.info("Generated questions are saved as: {}".format(output_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default='copa', type=str)
    parser.add_argument("--data_dir", default='../dataset/copa', type=str,
                        help="The input data dir. Should contain the .jsonl files for the task.")
    parser.add_argument("--input_file", default='dev.jsonl', type=str,
                        help="The input file.")
    parser.add_argument("--max_keyphrase_num", default=3, type=int,
                        help="Number of maximum keyphrases to extract.")
    parser.add_argument("--options_file", default='./elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json', type=str,
                        help="The options file of elmo.")
    parser.add_argument("--weight_file", default='./elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', type=str,
                        help="The input data dir. Should contain the .jsonl files for the task.")
    parser.add_argument("--corenlp_file", default='./corenlp/stanford-corenlp-4.2.2', type=str,
                        help="Stanford CoreNLP tool")
    parser.add_argument("--word_weight_file", default='auxiliary_data/enwiki_vocab_min200.txt', type=str)
    parser.add_argument("--extract_choices", action='store_true',
                        help="Whether to extract keyphares from choices.")



    args = parser.parse_args()
    args.is_training = False
    args.elmo_layers_weight = [0.0, 1.0, 0.0]
    args.output_file = os.path.splitext(args.input_file)[0] + '_with_{}_keyphrases.jsonl'.format(args.max_keyphrase_num)
    if args.extract_choices:
        args.output_file = os.path.splitext(args.input_file)[0] + '_with_{}_keyphrases_choices.jsonl'.format(args.max_keyphrase_num)



    ELMO = word_emb_elmo.WordEmbeddings(args.options_file, args.weight_file, cuda_device=0)
    SIF = sent_emb_sif.SentEmbeddings(ELMO, weightfile_pretrain=args.word_weight_file,
                                      weightfile_finetune=args.word_weight_file, lamda=1.0)
    extractor = StanfordCoreNLP(args.corenlp_file,
                               quiet=True)  # download from https://stanfordnlp.github.io/CoreNLP/

    processor = PROCESSOR_CLASSES[args.task_name.lower()](args)
    examples, fields = processor.load_examples(args.input_file, return_fields=True)

    extract(args, examples, fields, extractor, SIF)


if __name__ == '__main__':
    main()








