from typing import Dict, List, Tuple
from transformers import Trainer
from torch.utils.data.dataset import Dataset


class MultiEvalTrainer(Trainer):
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, epoch_end=False):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(epoch_end=epoch_end)
            self._report_to_hp_search(trial, epoch, metrics)

            for (lang, dataset) in self.eval_datasets:
                self.evaluate(eval_dataset=dataset, metric_key_prefix=lang + "_eval")

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics, epoch_end=epoch_end)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def add_eval_datasets(self, datasets: List[Tuple[str, Dataset]]):
        self.eval_datasets = datasets

if __name__ == "__main__":
    trainer = MultiEvalTrainer()