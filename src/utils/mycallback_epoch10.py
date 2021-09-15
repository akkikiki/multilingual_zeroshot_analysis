from transformers.trainer_callback import TrainerCallback
import os
import torch
import warnings


class MyCallbackEpoch10(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    def __init__(self):
        self.count = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        args.output_dir = self.output_dir
        assert os.path.exists(os.path.join(args.output_dir, "epoch"))
        return control
 
    def on_epoch_end(self, args, state, control, **kwargs):
        self.count += 1
        if self.count % 10 == 0:
            control.should_save = True
            control.should_evaluate = True
            args.output_dir = self.epoch_output_dir
            self.count = 0
        return control
    def set_epoch_output(self, output_dir):
        self.output_dir = output_dir
        self.epoch_output_dir = os.path.join(output_dir, "epoch")
        #if state.is_world_process_zero:
        #    state.save_to_json(os.path.join(args.output_dir, "epoch/trainer_state.json"))
        #    torch.save(optimizer.state_dict(), os.path.join(args.output_dir,
        #                                                         "epoch/optimizer.pt"))
        #    with warnings.catch_warnings(record=True) as caught_warnings:
        #        torch.save(lr_scheduler.state_dict(), os.path.join(args.output_dir,
        #                                                                "epoch/scheduler.pt"))
        #    # reissue_pt_warnings(caught_warnings)
