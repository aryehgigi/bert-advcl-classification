import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import BertModel, BertTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils import compute_metrics, AdvclProcessor, convert_examples_to_features
from config import Config
from model import AdvclTransformer


logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config, train_dataset, model, tokenizer):
    """ Train the model """
    config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.train_batch_size)

    if config.max_steps > 0:
        t_total = config.max_steps
        config.num_train_epochs = config.max_steps // (
            len(train_dataloader) // config.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=config.warmup_steps, t_total=t_total)

    # if fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=FP16_OPT_LEVEL)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                config.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                config.train_batch_size * config.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(config.num_train_epochs), desc="Epoch", disable=False)
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(config.seed)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(config.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'args_indices': batch[3],
                      'labels':      batch[4],
                      }

            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]
            if config.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % config.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if config.max_steps > 0 and global_step > config.max_steps:
                epoch_iterator.close()
                break
        if config.max_steps > 0 and global_step > config.max_steps:
            train_iterator.close()
            break
    return global_step, tr_loss / global_step


def evaluate(config, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = config.output_dir

    results = {}
    eval_dataset = load_and_cache_examples(config, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    config.eval_batch_size = config.per_gpu_eval_batch_size * max(1, config.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=config.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", config.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(config.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'args_indices': batch[3],
                      'labels':      batch[4],
                      }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    # output_eval_file = "eval/sem_res.txt"
    # with open(output_eval_file, "w") as writer:
    #     for key in range(len(preds)):
    #         writer.write("%d\t%s\n" %
    #                      (key+8001, str(RELATION_LABELS[preds[key]])))
    return result


def load_and_cache_examples(config, tokenizer, evaluate=False):
    processor = AdvclProcessor()
    if evaluate:
        examples = processor.get_dev_examples(config.data_dir)
    else:
        examples = processor.get_train_examples(config.data_dir)

    features = convert_examples_to_features(examples, tokenizer)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_args_indices = torch.tensor([f.args_indices for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_args_indices, all_label_ids)
    return dataset


def main():
    ### parsing aruments
    parser = ArgumentParser(
        description="BERT for ADVCL classification")
    parser.add_argument('--config', dest='config')
    args = parser.parse_args()
    config = Config(args.config)

    
    # Setup CUDA, GPU
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # Set seed
    set_seed(config.seed)

    # Prepare task
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower)
    model = BertModel.from_pretrained(config.pretrained_model_name)
    model = AdvclTransformer(model)
    model.to(device)

    # Training
    train_dataset = load_and_cache_examples(config, tokenizer, evaluate=False)
    global_step, tr_loss = train(config, train_dataset, model, tokenizer)
    logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # Create output directory if needed
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    logger.info("Saving model checkpoint to %s", config.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(config, os.path.join(config.output_dir, 'training_config.bin'))

    # # Load a trained model and vocabulary that you have fine-tuned
    # model = BertForSequenceClassification.from_pretrained(
    #     config.output_dir)
    # tokenizer = BertTokenizer.from_pretrained(
    #     config.output_dir, do_lower_case=True, additional_special_tokens=additional_special_tokens)
    # model.to(config.device)

    # Evaluation
    result = evaluate(config, model, tokenizer)

    return result


if __name__ == "__main__":
    main()
