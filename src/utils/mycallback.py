from transformers.trainer_callback import TrainerCallback
import os
import torch
import warnings


class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    def on_epoch_begin(self, args, state, control, **kwargs):
        args.output_dir = self.output_dir
        assert os.path.exists(os.path.join(args.output_dir, "epoch"))
        return control
 
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        control.should_evaluate = True
        args.output_dir = self.epoch_output_dir
        return control

    def set_epoch_output(self, output_dir):
        self.output_dir = output_dir
        self.epoch_output_dir = os.path.join(output_dir, "epoch")
