
import abc
import os
import sys
import tqdm
import torch
from torch.utils.data import DataLoader
from typing import Callable, Any
from pathlib import Path
from train_results import BatchResult, EpochResult, FitResult

class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, scheduler, device='cpu'):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, **kw) -> FitResult:

        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """

        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        epochs_so_far = 0
        checkpoint_filename=None
        if checkpoints is not None:
            checkpoint_filename = f'{checkpoints}.pt'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename,
                                         map_location=self.device)
                best_acc = saved_state.get('best_acc', best_acc)
                epochs_without_improvement =\
                    saved_state.get('ewi', epochs_without_improvement)
                epochs_so_far = saved_state.get('esf', epochs_so_far)
                actual_num_epochs += epochs_so_far
                self.model.load_state_dict(saved_state['model_state'])

        if epochs_so_far == num_epochs:
            batches = kw.get("max_batches") if "max_batches" in kw else None
            test_res = self.test_epoch(dl_test, max_batches=batches)
            test_loss.append(sum(test_res.losses) / len(test_res.losses))
            test_acc.append(test_res.accuracy)

        for epoch in range(epochs_so_far, num_epochs):
            save_checkpoint = True
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

            #  Train & evaluate for one epoch:
            # - Use the train/test_epoch methods.
            # - Save losses and accuracies in the lists above.
            # - Implement early stopping. This is a very useful and
            #   simple regularization technique that is highly recommended.
            batches = None
            if "max_batches" in kw:
                batches = kw.get("max_batches")

            actual_num_epochs += 1
            train_res = self.train_epoch(dl_train, verbose=verbose, max_batches=batches)
            test_res = self.test_epoch(dl_test, verbose=verbose, max_batches=batches)
            train_loss.append(sum(train_res.losses) / len(train_res.losses))
            train_acc.append(train_res.accuracy)
            test_loss.append(sum(test_res.losses) / len(test_res.losses))
            test_acc.append(test_res.accuracy)
            if early_stopping is not None and len(test_loss) >= 2:
                if test_loss[-1] >= test_loss[-2]:
                    epochs_without_improvement += 1
                    if epochs_without_improvement == early_stopping:
                        break
                else:
                    epochs_without_improvement = 0

            best_acc = max(best_acc if best_acc is not None else 0, test_res.accuracy)

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(best_acc=best_acc,
                                   ewi=epochs_without_improvement,
                                   model_state=self.model.state_dict(),
                                   esf=epoch+1,
                                   fit_result=FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc))

                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at epoch {epoch+1}')

            if post_epoch_fn:
                post_epoch_fn(epoch, train_res, test_res, verbose)

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)


class SRMTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, scheduler, device=None):
        super().__init__(model, loss_fn, optimizer, scheduler, device)

    def train_epoch(self, dl_train: DataLoader, **kw):
        self.h=None
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        self.h = None
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        X = X.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)

        #  Train the SRM model on one batch of data.
        # - Forward pass
        # - Calculate total loss over sequence
        # - Backward pass (BPTT)
        # - Update params
        # - Calculate number of correct char predictions

        self.optimizer.zero_grad()
        scores= self.model.forward(X)
        loss = self.loss_fn.forward(scores, y)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        y_hat = torch.argmax(scores, dim=1)
        num_correct = torch.sum(y_hat == y)
        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)

        with torch.no_grad():
            # Evaluate the SRM model on one batch of data.
            # - Forward pass
            # - Loss calculation
            # - Calculate number of correct predictions
            scores = self.model.forward(x)
            loss = self.loss_fn.forward(scores, y)
            y_hat = torch.argmax(scores, dim=1)
            num_correct = torch.sum(y_hat == y)

        return BatchResult(loss.item(), num_correct.item())