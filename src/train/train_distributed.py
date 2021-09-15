import torch, os, sys
import transformers
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

from src.utils.wikibib_dataset import WikiBibDataset
from src.utils.myiterable_dataset import MyIterableDataset
from src.utils.multieval_trainer import MultiEvalTrainer
from src.utils.mycallback import MyCallback
from src.utils.mycallback_epoch10 import MyCallbackEpoch10

import argparse, logging


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_train_directory', type=str, help='directory containing files for the source language')
    parser.add_argument('--source_dev_directory', type=str, help='directory containing files for the source language (dev set)')
    parser.add_argument('--target_train_directory', type=str, help='directory containing files for the target language')
    parser.add_argument('--source_lang', type=str)
    parser.add_argument('--target_lang', type=str)

    parser.add_argument('--save_at_end', action='store_const', const=True, help='save model after training', default=True)
    parser.add_argument('--output_directory', type=str)

    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--wandb', action='store_const', const=True, default=False)
    parser.add_argument('--test', action='store_const', const=True, default=False)
    parser.add_argument('--remove_underscore', action='store_const', const=True, default=False)

    parser.add_argument('--continue_training', type=str)
    parser.add_argument('--load_from_checkpoint', type=str, help='initialize model from checkpoint')
    parser.add_argument('--tokenizer_path', type=str, help='path to tokenizer', default='xlm-roberta-base')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--train_sampler', type=str, default='baseline', help='sampling scheme for training data. default: baseline')
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument('--num_gpus', type=int)
    parser.add_argument('--num_langs', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=2500)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--full_size_corpus', action='store_const', const=True, default=False)
    parser.add_argument('--no_optimizer_scheduler', action='store_const', const=True, default=False)
    parser.add_argument('--no_save_every_epoch', action='store_const', const=True, default=False)
    parser.add_argument('--no_early_stop', action='store_const', const=True, default=False)
    parser.add_argument('--epochs_trained_zero', action='store_const', const=True, default=False)
    parser.add_argument('--save_every_10epochs', action='store_const', const=True, default=False)
    parser.add_argument('--best_model_checkpoint_none', action='store_const', const=True, default=False)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--id2lang', nargs='+', type=str, default=["en", "ru", "zh", "ar", "hi"])
    parser.add_argument('--langs_to_use', nargs='+', type=str, default=["en", "ru", "zh", "ar", "hi"])
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=-1)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if len(args.langs_to_use) < 5:
        logging.info("num_langs should be set to larger than 5 if langs to use is less than 5")
        logging.info("Make sure to set to the maximum if randomly permuting the order")
        assert args.num_langs >= 5

    if args.wandb:
        import wandb
        wandb.init(project=args.experiment_name)

    logging.root.handlers = []
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("{}/{}.log".format(args.log_dir, args.experiment_name)),
            logging.StreamHandler()
        ]
    )

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.tokenizer_path, max_len=args.max_seq_len, padding='max_length')

    logging.info('Given Args')
    logging.info(args)



    if args.load_from_checkpoint:
        logging.info('Continuing training from {}'.format(args.load_from_checkpoint))
        model = XLMRobertaForMaskedLM.from_pretrained(args.load_from_checkpoint)
    else:
        logging.info('Initializing new random XLM-R model')
        config = XLMRobertaConfig.from_pretrained('xlm-roberta-base')
        model = XLMRobertaForMaskedLM(config=config)

    logging.info('Model Config')
    logging.info(model.config)

    logging.info('Loading Files')
    if args.full_size_corpus:
        assert os.path.isfile(args.source_train_directory)
        from torch.utils.data import BufferedShuffleDataset
        temp = MyIterableDataset(tokenizer=tokenizer,
                                 max_len=args.max_seq_len,
                                 file_path=args.source_train_directory,
                                 )
        train_dataset = BufferedShuffleDataset(temp, 1024)
    else:
        train_dataset = WikiBibDataset(tokenizer=tokenizer,
                                       source_directory=args.source_train_directory,
                                       target_directory=args.target_train_directory,
                                       train_sampler=args.train_sampler,
                                       max_len=args.max_seq_len,
                                       source_lang = args.source_lang,
                                       target_lang = args.target_lang,
                                       langs_to_use = args.langs_to_use,
                                       id2lang = args.id2lang,
                                       remove_underscore = args.remove_underscore,
                                       num_langs=args.num_langs,
                                       seed=args.seed)

    if args.test:
        dev_dataset = None
    else:
        dev_dataset = WikiBibDataset(tokenizer=tokenizer,
                                     source_directory=args.source_dev_directory,
                                     target_directory=args.target_train_directory,
                                     train_sampler=args.train_sampler,
                                     max_len=args.max_seq_len,
                                     source_lang=args.source_lang,
                                     langs_to_use = args.langs_to_use,
                                     id2lang = args.id2lang,
                                     target_lang=args.target_lang,
                                     remove_underscore = args.remove_underscore,
                                     num_langs=args.num_langs)  # all languages
    lang_dev_datasets = []
    id2lang = args.id2lang

    for i in range(5):
        file_path = args.source_dev_directory + f"/source_packedNone.txt_{i}.txt"
        lang_dev_dataset = WikiBibDataset(tokenizer=tokenizer,
                                          source_directory=None,
                                          target_directory=args.target_train_directory,
                                          train_sampler=args.train_sampler,
                                          max_len=args.max_seq_len,
                                          source_lang=args.source_lang,
                                          target_lang=args.target_lang,
                                          num_langs=0,
                                          file_path=file_path,
                                          remove_underscore = args.remove_underscore,
                                          )
        lang_dev_datasets.append((id2lang[i], lang_dev_dataset))
    dev_dataset_all = WikiBibDataset(tokenizer=tokenizer,
                                     source_directory=args.source_dev_directory,
                                     target_directory=args.target_train_directory,
                                     train_sampler=args.train_sampler,
                                     max_len=args.max_seq_len,
                                     source_lang=args.source_lang,
                                     target_lang=args.target_lang,
                                     remove_underscore=args.remove_underscore,
                                     num_langs=5)  # all languages
    lang_dev_datasets.append(("all", dev_dataset_all))

    if (not args.full_size_corpus) and train_dataset.source_examples:
        logging.info(train_dataset.source_examples[0])


    collator = DataCollatorForLanguageModeling(
        mlm=True,
        tokenizer=tokenizer,
        mlm_probability=0.15
    )
    
    effective_bs = args.gradient_accumulation_steps * args.batch_size * args.num_gpus
    if args.warmup_steps != -1:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int((args.epochs * (train_dataset.__len__() // (effective_bs))) * .01)

    place_model_on_device = not args.continue_training

    training_args = TrainingArguments(
        output_dir=args.output_directory,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=2500,
        save_total_limit=1,
        prediction_loss_only=True,
        logging_strategy="steps",
        save_strategy="steps",
        evaluation_strategy="steps",
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        dataloader_num_workers=1,
        local_rank=args.local_rank,
        disable_tqdm= True,
        max_steps=args.max_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    training_args.epochs_trained_zero = args.epochs_trained_zero
    training_args.best_model_checkpoint_none = args.best_model_checkpoint_none
    training_args.no_optimizer_scheduler = args.no_optimizer_scheduler

    logging.info('Training Args')
    logging.info(training_args)
    callbacks = []

    if not args.no_save_every_epoch:
        if args.save_every_10epochs:
            logging.info("Saving every 10 epochs")
            my_callback = MyCallbackEpoch10()
        else:
            logging.info("Saving every epoch")
            my_callback = MyCallback()
        my_callback.set_epoch_output(args.output_directory)
        callbacks.append(my_callback)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.01,
    )
    if not args.no_early_stop:
        logging.info("early stopping triggered")
        callbacks.append(early_stopping_callback)

    trainer = MultiEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.add_eval_datasets(lang_dev_datasets)

    if args.continue_training:
        trainer.train(resume_from_checkpoint=args.continue_training)
        print("continue")
    else:
        trainer.train()

    if args.save_at_end:
        model.save_pretrained(args.output_directory + 'final/')
        print(trainer.evaluate())
