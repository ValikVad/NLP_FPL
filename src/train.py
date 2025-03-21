import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os
from tqdm import tqdm 


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./checkpoints",
    ):
        """
        Initialize the Trainer class.

        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            criterion (nn.Module): Loss function.
            optimizer (optim.Optimizer): Optimizer for training.
            scheduler (Optional[optim.lr_scheduler._LRScheduler]): Learning rate scheduler (optional).
            device (str): Device to use ('cuda' or 'cpu').
            save_dir (str): Directory to save model checkpoints.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir

        # Ensure the save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize metrics
        self.best_val_loss = float("inf")

    def train_epoch(self) -> Dict[str, float]:
        """
        Perform one epoch of training.

        Returns:
            Dict[str, float]: Training metrics (e.g., loss).
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for inputs, targets in tqdm(self.train_loader):
            if not  isinstance(inputs, dict):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_samples += 1

        avg_loss = total_loss / total_samples
        return {"loss": avg_loss}

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the validation set.

        Returns:
            Dict[str, float]: Validation metrics (e.g., loss).
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader):
                if not  isinstance(inputs, dict):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Update metrics
                total_loss += loss.item() 
                total_samples += 1

        avg_loss = total_loss / total_samples
        return {"loss": avg_loss}

    def train(self, num_epochs: int, checkpoint_name: str = "best_model.pth") -> None:
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
            checkpoint_name (str): Name of the file to save the best model checkpoint.
        """
        for epoch in tqdm(range(1, num_epochs + 1)):
            print(f"Epoch {epoch}/{num_epochs}")

            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()

            # Log metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")

            # Save the best model based on validation loss
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"New best model saved at {checkpoint_path}")

            # Step the learning rate scheduler
            if self.scheduler:
                self.scheduler.step()

        print("Training complete.")

