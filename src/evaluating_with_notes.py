from tqdm import tqdm, trange
import torch
import logging
import argparse
import numpy as np
import random
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score
import os
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


from processors import CopaProcessor, SocialIQAProcessor, ROCStoryProcessor

logger = logging.getLogger(__name__)


PROCESSOR_CLASSES = {
    'copa': CopaProcessor,
    'socialiqa': SocialIQAProcessor,
    'rocstory': ROCStoryProcessor
}

MODEL_CLASSES = [
    'bert',
    'albert',
    'ctrl',
    'distilbert',
    'gpt2',
    'distilgpt2',
    'openai-gpt',
    'roberta',
    't5',
    'xlnet',
    'xlm'
]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def compute_losses(args, logits, inputs_ids, prefixes_lengths, tokenizer):
    """
    :param args
    :param logits: FloatTensor, shape of (batch_size, seq_len, vocab_size)
    :param inputs_ids: LongTensor, shape of (batch_size, seq_len)
    :param prefixes_lengths: LongTensor, shape of (batch_size)
    :param tokenizer
    :return: FloatTensor, shape of (batch_size)
    """
    batch_size, seq_len, vocab_size = logits.shape
    loss_fct = CrossEntropyLoss()
    shift_labels = inputs_ids[:, 1:].contiguous().view(-1).clone()
    shift_labels[shift_labels == tokenizer.pad_token_id] = -100
    shift_labels = shift_labels.reshape(batch_size, -1)
    shift_logits = logits[:, :-1, :].contiguous()  # shift left
    losses = torch.zeros([batch_size], device=args.device)
    for idx in range(batch_size):
        if args.use_part_loss:
            # Only count scores for last part
            prediction_scores = shift_logits[idx, prefixes_lengths[idx]:, :]  # shape of (valid_length, vocab_size)
            loss = loss_fct(prediction_scores, shift_labels[idx, prefixes_lengths[idx]:])
        else:
            prediction_scores = shift_logits[idx, :, :]  # shape of (valid_length, vocab_size)
            loss = loss_fct(prediction_scores, shift_labels[idx, :])
        losses[idx] = loss
    return losses


class QAInput(object):
    def __init__(self,
                 choices_prompts_inputs=None,
                 context_prompts_inputs=None,
                 choices_prefixes_lengths=None,
                 context_prefixes_lengths=None):
        """
        :param choices_prompts_inputs: list of [dict] * num_choices, dict.input_ids: shape of (num_notes, seq_len)
        :param context_prompts_inputs:  dict, dict.input_ids: shape of (num_choices, seq_len)
        :param choices_prefixes_lengths: list of [item] * num_choices, item: shape of (notes)
        :param context_prefixes_lengths:  LongTensor, shape of (num_choices)
        """
        self.choices_prompts_inputs = choices_prompts_inputs
        self.context_prompts_inputs = context_prompts_inputs
        self.choices_prefixes_lengths = choices_prefixes_lengths
        self.context_prefixes_lengths = context_prefixes_lengths


def convert_example_to_inputs(args, example, processor, tokenizer):
    feature = processor.convert_example_to_feature(example)
    all_prompts_for_choices_inputs = []
    all_prefixes_for_choices_lengths = []
    for choice_idx, prompt in enumerate(feature.prompts_for_choices):
        prompt_for_choices_with_notes = []
        if len(example.notes) == 0 or example.notes is None or not args.take_notes:
            example.notes = ['']
        for note in example.notes[: args.max_notes_num]:
            if args.attaching_side == 'right':
                prompt_for_choices_with_notes.append(prompt + ' ' + note)
            else:
                prompt_for_choices_with_notes.append(note + ' ' + prompt)
        prompts_with_notes_inputs = tokenizer(prompt_for_choices_with_notes, return_tensors='pt', padding=True,
                                              max_length=args.max_seq_length, truncation=True)
        prompts_with_notes_inputs = prompts_with_notes_inputs.to(args.device)
        all_prompts_for_choices_inputs.append(prompts_with_notes_inputs)

        choice_len = len(tokenizer.tokenize(feature.choices[choice_idx]))
        prompts_len = [len(tokenizer.tokenize(prompt)) for prompt in prompt_for_choices_with_notes]
        prompts_len = torch.tensor(prompts_len, dtype=torch.long, device=args.device)
        prefixes_len = prompts_len - choice_len  # shape of (num_notes)
        prefixes_len = prefixes_len + 1 if 'gpt' not in args.model_type.lower() else prefixes_len
        all_prefixes_for_choices_lengths.append(prefixes_len)
    if not args.use_score_x:
        return QAInput(choices_prompts_inputs=all_prompts_for_choices_inputs,
                       choices_prefixes_lengths=all_prefixes_for_choices_lengths
                       )
    else:
        prompts_for_context_inputs = tokenizer(feature.prompts_for_context, return_tensors='pt', padding=True,
                                               max_length=args.max_seq_length, truncation=True)
        prompts_for_context_inputs = prompts_for_context_inputs.to(args.device)

        cont_len = len(tokenizer.tokenize(feature.context))
        context_prompts_len = [len(tokenizer.tokenize(prompt)) for prompt in feature.prompts_for_context]
        context_prompts_len = torch.tensor(context_prompts_len, dtype=torch.long, device=args.device)
        context_prefixes_len = context_prompts_len - cont_len
        context_prefixes_len = context_prefixes_len + 1 \
            if 'gpt' not in args.model_type.lower() else context_prefixes_len

        return QAInput(all_prompts_for_choices_inputs, prompts_for_context_inputs,
                       all_prefixes_for_choices_lengths, context_prefixes_len
                       )


def evaluate(args, qa_inputs, model, tokenizer):
    """Evaluating"""
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Evaluating
    logger.info("***** Running evaluate *****")
    logger.info("  Num examples = %d", args.num_examples)

    model.eval()
    final_results = []
    for sample_idx, qa_input in enumerate(tqdm(qa_inputs, desc='Evaluating')):
        num_choices = len(qa_input.choices_prompts_inputs)
        opt_losses = torch.zeros([num_choices], dtype=torch.float, device=args.device)
        with torch.no_grad():
            for choice_idx, choice_inputs in enumerate(qa_input.choices_prompts_inputs):
                b_s = 0
                b_e = args.batch_size
                curr_choice_losses = torch.tensor([], dtype=torch.float, device=args.device)
                curr_choice_prefixes_lengths = qa_input.choices_prefixes_lengths[choice_idx]
                while b_s < choice_inputs.input_ids.shape[0]:
                    batch_inputs_ids = choice_inputs.input_ids[b_s: b_e]
                    batch_attention_mask = choice_inputs.attention_mask[b_s: b_e]
                    outputs = model(input_ids=batch_inputs_ids,
                                    attention_mask=batch_attention_mask)
                    logits = outputs.logits
                    batch_losses = compute_losses(args, logits, batch_inputs_ids,
                                                  curr_choice_prefixes_lengths[b_s: b_e],
                                                  tokenizer)
                    curr_choice_losses = torch.cat((curr_choice_losses, batch_losses), dim=0)
                    b_s = b_e
                    b_e += args.batch_size
                if args.scale_type == 'mean':
                    opt_losses[choice_idx] = curr_choice_losses.mean()
                else:
                    opt_losses[choice_idx] = curr_choice_losses.min()

            opt_scores = torch.nn.Softmax(dim=0)(opt_losses.mean() - opt_losses)

            if args.use_score_x:
                b_s = 0
                b_e = args.batch_size
                cont_losses = torch.tensor([], dtype=torch.float, device=args.device)
                while b_s < num_choices:
                    cont_batch_input_ids = qa_input.context_prompts_inputs.input_ids[b_s: b_e]
                    cont_batch_attention_mask = qa_input.context_prompts_inputs.attention_mask[b_s: b_e]

                    cont_outputs = model(input_ids=cont_batch_input_ids,
                                         attention_mask=cont_batch_attention_mask)
                    cont_logits = cont_outputs.logits
                    cont_batch_losses = compute_losses(args, cont_logits, cont_batch_input_ids,
                                                       qa_input.context_prefixes_lengths[b_s: b_e],
                                                       tokenizer)
                    cont_losses = torch.cat((cont_losses, cont_batch_losses), dim=0)
                    b_s = b_e
                    b_e += args.batch_size
                cont_scores = torch.nn.Softmax(dim=0)(cont_losses.mean() - cont_losses)

                final_scores = 0.5*opt_scores + 0.5*cont_scores
            else:
                final_scores = opt_scores
            final_result = final_scores.argmax().item()
            final_results.append(final_result)

    return final_results

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default='gpt2', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES))
    parser.add_argument("--model_name_or_path", default='../pretrained_model/gpt2-medium', type=str, required=False)
    parser.add_argument("--output_dir", default='./outputs', type=str, required=False,
                        help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--task_name", default='copa', type=str)
    parser.add_argument("--data_dir", default='./dataset/copa', type=str,
                        help="The input data dir. Should contain the .jsonl files for the task.")
    parser.add_argument("--eval_file", default="new_dev_with_3_keyphrases_32_notes_gpt2-medium.jsonl", type=str,
                        help="The input training file.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="Maximum tokens of input sequence.")
    parser.add_argument("--take_notes", action='store_true')
    parser.add_argument("--max_notes_num", default=32, type=int)
    parser.add_argument("--use_score_x", action='store_true')
    parser.add_argument("--use_part_loss", action='store_true')
    parser.add_argument("--batch_size_per_gpu", default=4, type=int,
                        help="Batch size on each GPU.")
    parser.add_argument("--attaching_side", default='left', type=str,
                        help="Where to attach note with the statement.")
    parser.add_argument("--scale_type", default='min', type=str,
                        help="Scaled function")
    parser.add_argument("--calculate_accuracy", default=True, type=bool,
                        help="Whether to calculate evaluation accuracy.")
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
    if not args.take_notes:
        args.max_notes_num = 1

    output_fname = '{}_predictions.txt'.format(os.path.splitext(args.eval_file)[0])

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    output_file = os.path.join(args.output_dir, output_fname)
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
    args.batch_size = args.batch_size_per_gpu * args.n_gpu

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args)

    # Load evaluating model
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.is_decoder = True
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Evaluating parameters %s", args)

    # Prepare data
    args.is_training = False
    args.task_name = args.task_name.lower()
    processor = PROCESSOR_CLASSES[args.task_name](args)
    examples = processor.load_examples(args.eval_file)
    args.num_examples = len(examples)
    qa_inputs = []
    for example in tqdm(examples, desc='Converting examples to inputs'):
        qa_inputs.append(convert_example_to_inputs(args, example, processor, tokenizer))

    # Do evaluating
    predictions = evaluate(args, qa_inputs, model, tokenizer)

    if args.calculate_accuracy and examples[0].label is not None:
        labels = []
        for example in examples:
            labels.append(example.label)

        acc = accuracy_score(labels, predictions)
        logger.info("Evaluating accuracy: {}%".format(acc * 100))

        with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
            f.write("*****On: {}***** \n".format(args.eval_file))
            f.write("PrLM: {}\n".format(os.path.basename(args.model_name_or_path)))
            f.write("Evaluating accuracy: {}%".format(acc * 100))
    with open(output_file, 'w') as f_out:
        for prediction in predictions:
            f_out.write(str(prediction)+'\n')
            f_out.flush()
        logger.info("The predictions are written in file: {}".format(output_file))


if __name__ == "__main__":
    main()







