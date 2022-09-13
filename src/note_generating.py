import torch
import logging
import string
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
from processors import (CopaProcessor, SocialIQAProcessor, ROCStoryProcessor)
from torch.nn import CrossEntropyLoss
import random
import re
import json
import argparse
import os
from nltk.stem import WordNetLemmatizer
from multiprocessing.dummy import Pool
wnl = WordNetLemmatizer()

logger = logging.getLogger(__name__)

MODEL_CLASSES = [
    'distilgpt2',
    'gpt2',
    'openai-gpt',
    'roberta',
    'bert',
    'albert',
    'bart',
    'xlnet',
    't5'
]

PROCESSOR_CLASSES = {
    'copa': CopaProcessor,
    'socialiqa': SocialIQAProcessor,
    'rocstory': ROCStoryProcessor
}


COMMON_WORDS = ['man', 'woman', 'boy', 'girl', 'men', 'women', 'people', 'person', 'one', 'everyone', 'anyone', 'someone'
                'something', 'everything', 'anything']


class NoteInput(object):
    def __init__(self, context, note_prefixes, insert_positions):
        self.context = context
        self.note_prefixes = note_prefixes
        self.insert_positions = insert_positions


def convert_examples_to_inputs(args, examples, processor):
    all_inputs = []
    for example in examples:
        context = example.context
        prefixes = []
        insert_positions = []
        key_phrases = example.key_phrases[: args.max_keyphrase_num]
        for phrase in key_phrases:
            # phrase: [text, [s, e], type]
            if phrase[0].lower() in COMMON_WORDS:
                continue
            if phrase[-1] == 'NP':
                curr_prefixes = [tmp.replace('[NP]', phrase[0]) for tmp in processor.note_prefixes['NP']]
                prefixes += curr_prefixes
                insert_positions.append([phrase[1][1]] * len(curr_prefixes))
            elif phrase[-1] == 'VP':
                phrase_tokens = phrase[0].split(' ')
                phrase_tokens[0] = wnl.lemmatize(phrase_tokens[0], pos='v')
                phrase_tokens[0] = phrase_tokens[0][:-1] + 'ing' if phrase_tokens[0][-1] == 'e' else phrase_tokens[0] + 'ing'
                NVP_phrase = ' '.join(phrase_tokens)
                curr_prefixes = [tmp.replace('[VP]', NVP_phrase) for tmp in processor.note_prefixes['VP']]
                prefixes += curr_prefixes
                insert_positions.append(phrase[1][1] * len(curr_prefixes))
            elif phrase[-1] == 'NNP':
                curr_prefixes = [tmp.replace('[NNP]', phrase[0]) for tmp in processor.note_prefixes['NNP']]
                prefixes += curr_prefixes
                insert_positions.append(phrase[1][1] * len(curr_prefixes))
            elif phrase[-1] == 'ST':
                curr_prefixes = [phrase[0] + ' means that']
                prefixes += curr_prefixes
                insert_positions.append(phrase[1][1] * len(curr_prefixes))
        all_inputs.append(NoteInput(context, prefixes, insert_positions))
    return all_inputs


def generate(args, inputs, fields, model, tokenizer, output_file):
    """Generate notes with pre-trained model"""
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    stop_token = '.'
    eos_token_ids = tokenizer.encode("{} {} {}".format(stop_token, tokenizer.eos_token, tokenizer.pad_token))

    # Generating
    logger.info("***** Running generate *****")
    logger.info("  Num examples = %d", args.num_examples)
    logger.info("  Max generated length = %d", args.max_gen_length)
    logger.info("  Top p for sampling = %f", args.top_p)

    model.eval()
    for idx, input in enumerate(tqdm(inputs, 'Generating notes')):
        with torch.no_grad():
            b_s = 0
            b_e = args.batch_size
            prefixes = input.note_prefixes
            gen_notes = []
            while b_s < len(prefixes):
                prefix_batch = prefixes[b_s: b_e]
                note_prefixes_txt = [input.context + ' ' + p for p in prefix_batch]
                note_prefixes = tokenizer(note_prefixes_txt, return_tensors='pt', padding=True, truncation=True,
                                          max_length=args.max_seq_length)
                note_prefixes = note_prefixes.to(args.device)
                batch_size, prefix_len = note_prefixes.input_ids.shape

                curr_gen_notes = []
                time = 0
                while len(curr_gen_notes) < 1.5 * batch_size and time < args.max_gen_time:
                    time += 1
                    batch_outputs = model.generate(**note_prefixes, max_length=args.max_gen_length + prefix_len,
                                                   do_sample=True, temperature=args.temperature, num_return_sequences=args.num_once_gen,
                                                   top_p=args.top_p, top_k=args.top_k, eos_token_ids=eos_token_ids,
                                                   repetition_penalty=args.repetition_penalty, pad_token_id=tokenizer.pad_token_id)
                    if len(batch_outputs.shape) == 3:
                        batch_outputs = batch_outputs[0]
                    gen_sequences = batch_outputs[:, prefix_len:]  # shape of (batch_size*num_once_gen, max_gen_length)
                    curr_gen_notes += get_generative_text_with_batch(args, tokenizer, gen_sequences, prefix_batch)
                gen_notes = list(set(gen_notes + curr_gen_notes))
                b_s = b_e
                b_e += args.batch_size
            if args.do_rethink:
                final_notes = rethink_notes(args, input.context, gen_notes, model, tokenizer, args.return_notes_num)
            else:
                return_notes_num = len(gen_notes) if args.return_notes_num > len(gen_notes) else args.return_notes_num
                selected_indices = random.sample(range(len(gen_notes)), return_notes_num)
                final_notes = [gen_notes[i] for i in selected_indices]
            fields[idx]['notes'] = ['None'] + final_notes

    with open(output_file, 'w') as f_out:
        for field in fields:
            f_out.write(json.dumps(field) + '\n')
            f_out.flush()
    logger.info("Generated notes are saved as: {}".format(output_file))


def compute_losses(logits, inputs_ids, pad_token_id):
    """
    :param logits: FloatTensor, shape of (batch_size, seq_len, vocab_size)
    :param inputs_ids: LongTensor, shape of (batch_size, seq_len)
    :param pad_token_id:
    :return:
    """
    batch_size, seq_len, vocab_size = logits.shape
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_labels = inputs_ids[:, 1:].contiguous().view(-1).clone()
    shift_labels[shift_labels == pad_token_id] = -100
    shift_logits = logits[:, :-1, :].contiguous()  # shift left
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    losses = loss_fct(shift_logits, shift_labels)  # shape of (batch_size * (seq_len -1))
    losses = losses.reshape(batch_size, -1).mean(dim=1)  # shape of (batch_size)
    return losses


def rethink_notes(args, context, notes, model, tokenizer, return_notes_num):
    """
    :param args:
    :param context: str
    :param notes: [str] * num_notes
    :param model:
    :param tokenizer:
    :param return_notes_num: int, Number of final selected notes
    :return:
    """
    return_notes_num = len(notes) if return_notes_num > len(notes) else return_notes_num
    tokenizer.padding_side = 'right'
    context_with_notes = [context + ' ' + note for note in notes]
    b_s = 0
    b_e = args.batch_size
    all_losses = torch.tensor([], dtype=torch.float, device=args.device)
    while b_s < len(notes):
        cwn_batch = context_with_notes[b_s: b_e]
        inputs = tokenizer(cwn_batch, return_tensors='pt', padding=True,
                           max_length=args.max_seq_length, truncation=True)
        inputs = inputs.to(args.device)
        outputs = model(**inputs)
        logits = outputs.logits  # shape of (batch_size, seq_len, vocab_size)
        batch_losses = compute_losses(logits, inputs.input_ids, tokenizer.pad_token_id)
        all_losses = torch.cat((all_losses, batch_losses), dim=0)
        b_s = b_e
        b_e += args.batch_size

    # all_losses: shape of (batch_size)
    selected_indices = all_losses.topk(return_notes_num, largest=False, sorted=True)[1]
    final_notes = [notes[i] for i in selected_indices]
    tokenizer.padding_side = 'left'
    return final_notes


def get_generative_text_with_batch(args, tokenizer, sequences, prefixes, min_len=2):
    """
    :param args
    :param tokenizer
    :param sequences: Raw output sequences, LongTensor of shape: (batch_size*num_once_gen, max_gen_length)
    :param prefixes: list of prefix_text, [str] * batch_size
    :param min_len: Minimum length of generative text
    :return:
    """
    all_prefixes = []
    for prefix in prefixes:
        all_prefixes += [prefix] * args.num_once_gen
    gen_texts = [tokenizer.decode(seq, clean_up_tokenization_spaces=True) for seq in sequences]
    final_results = []
    for text, prefix in zip(gen_texts, all_prefixes):
        if '\n' in text:
            text = text[: text.find('\n')].strip()
        if tokenizer.eos_token in text:
            text = text[: text.find(tokenizer.eos_token)].strip()
        if tokenizer.pad_token in text:
            text = text[: text.find(tokenizer.pad_token)].strip()
        text = text[: text.find('.') + 1].strip()
        if ',' in text:
            text = text[: text.find(',')].strip() + '.'
        text = re.sub(" +", " ", text).strip()
        if len(text.split(' ')) >= min_len:
            if ('still' in text or 'not' in text.lower()) and len(text.split(' ')) <= min_len+1:
                continue
            note = prefix + ' ' + text
            final_results.append(note.strip())
    # logger.info(final_results)
    return list(set(final_results))


def get_generative_text(tokenizer, gen_sequences, stop_token='.'):
    """
    :param tokenizer
    :param gen_sequences: Raw generated sequences, LongTensor of shape: (num_once_generating, max_seq_length)
    :param stop_token: End token of the generated question
    :return:
    """
    gen_raw_txts = [tokenizer.decode(seq, clean_up_tokenization_spaces=True) for seq in gen_sequences]
    gen_txts = [text[:text.find(stop_token)+1] for text in gen_raw_txts if stop_token in text]
    gen_txts = [txt.replace(tokenizer.eos_token, "") for txt in gen_txts]
    gen_txts = [txt.replace('\n', " ") for txt in gen_txts]
    gen_txts = [re.sub(" +", " ", txt).strip() for txt in gen_txts]
    gen_txts = [remove_punctuations(txt) for txt in gen_txts]
    gen_txts = [txt for txt in gen_txts if len(txt.split(' ')) > 2]
    return gen_txts


def remove_punctuations(s):
    punctuations = string.punctuation
    for c in '.-\'/':
        punctuations = punctuations.replace(c, '')
    s_chars = list(s)
    for i in range(len(s_chars)):
        s_chars[i] = '' if s_chars[i] in list(punctuations) else s_chars[i]
    return ''.join(s_chars)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default='gpt2', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES))
    parser.add_argument("--model_name_or_path", default='../pretrained_model/gpt2-medium', type=str, required=False)
    parser.add_argument("--output_dir", default='./dataset/copa', type=str, required=False,
                        help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--task_name", default='copa', type=str)
    parser.add_argument("--data_dir", default='./dataset/copa', type=str,
                        help="The input data dir. Should contain the .jsonl files for the task.")
    parser.add_argument("--input_file", default="new_dev_with_3_keyphrases.jsonl", type=str,
                        help="The input file.")
    parser.add_argument("--return_notes_num", default=32, type=int,
                        help="Batch size of per GPU.")
    parser.add_argument("--batch_size_per_gpu", default=2, type=int,
                        help="Batch size of per GPU.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="Max length of input sequence after tokenization. Over part will be truncated.")
    parser.add_argument("--max_gen_time", default=3, type=int,
                        help="Maximun time of generating.")
    parser.add_argument("--max_keyphrase_num", default=5, type=int)
    parser.add_argument("--num_once_gen", default=50, type=int,
                        help="How many sequences to generate at once.")
    parser.add_argument("--max_gen_length", default=15, type=int,
                        help="Max length of generated sequences.")
    parser.add_argument("--top_p", default=0.8, type=float,
                        help="top_p for sampling of sequence generation.")
    parser.add_argument("--top_k", default=0.0, type=float,
                        help="top_k for sampling of sequence generation.")
    parser.add_argument("--repetition_penalty", default=2.0, type=float,
                        help="Repetition penalty for generation.")
    parser.add_argument("--temperature", default=1.0, type=float,
                        help="Repetition penalty for generation.")
    parser.add_argument('--do_rethink', action='store_true',
                        help="Do notes rethinking.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--overwrite_output_file', action='store_true',
                        help="Overwrite the output file")
    parser.add_argument('--device_ids', default='0', type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids

    output_fname = "{}_{}_notes_{}.jsonl".format(
        os.path.splitext(args.input_file)[0],
        args.return_notes_num,
        os.path.basename(args.model_name_or_path)
        )
    if not args.do_rethink:
        output_fname = "{}_{}_notes_random_{}.jsonl".format(
            os.path.splitext(args.input_file)[0],
            args.return_notes_num,
            os.path.basename(args.model_name_or_path)
        )

    output_file = os.path.join(args.data_dir, output_fname)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if os.path.exists(output_file) and not args.overwrite_output_file:
        raise ValueError(
            "Output file ({}) already exists and is not empty. Use --overwrite_output_file to overcome.".format(
                output_file))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    args.batch_size = args.batch_size_per_gpu

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.is_decoder = True
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Generating parameters %s", args)

    # Prepare data
    args.is_training = False
    args.task_name = args.task_name.lower()
    processor = PROCESSOR_CLASSES[args.task_name](args)
    examples, fields = processor.load_examples(args.input_file, return_fields=True)
    args.num_examples = len(examples)
    inputs = convert_examples_to_inputs(args, examples, processor)

    # Do Generating
    generate(args, inputs, fields, model, tokenizer, output_file)


if __name__ == '__main__':
    main()