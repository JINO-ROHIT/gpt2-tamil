import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numpy as np

from layers import TamilGPT

class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: DataLoader, 
                 test_loader: DataLoader, 
                 loss: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 epochs: int, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir: str = 'model_checkpoints',
                 model_name: str = 'best_model'):
        """
        Initialize the Trainer with model, data loaders, loss function, optimizer, and training parameters.
        
        Args:
            model (nn.Module): The neural network model to train
            train_loader (DataLoader): DataLoader for training data
            test_loader (DataLoader): DataLoader for validation/test data
            loss (nn.Module): Loss function for training
            optimizer (torch.optim.Optimizer): Optimizer for model parameters
            epochs (int): Number of training epochs
            device (str, optional): Device to run training on. Defaults to cuda if available.
            checkpoint_dir (str, optional): Directory to save model checkpoints
            model_name (str, optional): Base name for saved model files
        """
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.model_name = model_name
        
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def _calculate_loss(self, inputs, targets):
        """
        Calculate loss for given inputs and targets.
        
        Args:
            inputs (torch.Tensor): Input tensors
            targets (torch.Tensor): Target tensors
        
        Returns:
            torch.Tensor: Calculated loss
        """
        logits = self.model(inputs)
        loss = self.loss(logits.flatten(0, 1), targets.flatten())
        return loss
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """
        Save model checkpoint if it's the best validation loss so far.
        
        Args:
            epoch (int): Current training epoch
            val_loss (float): Validation loss for current epoch
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
            checkpoint_filename = f"{self.model_name}_epoch{epoch+1}_loss{val_loss:.4f}.pth"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_loss
            }, checkpoint_path)
            
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
            
            self.best_model_path = checkpoint_path
            
            print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    def _train(self) -> None:
        """
        Train the model for specified number of epochs.
        Tracks training progress and periodically evaluates the model.
        """
        steps = 0
        for epoch in tqdm(range(self.epochs), desc="Training"):
            self.model.train()
            epoch_loss = 0.0
            
            for idx, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                loss = self._calculate_loss(inputs, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                steps += 1
                
                if steps % 100 == 0:
                    train_loss, validation_loss = self._evaluate()
                    print(f"Steps: {steps}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}")
            
            train_loss, validation_loss = self._evaluate()
            self._save_checkpoint(epoch, validation_loss)
            
            print(f"Epoch {epoch+1}/{self.epochs}, Average Train Loss: {epoch_loss/len(self.train_loader):.4f}")
    
    @torch.inference_mode()
    def _evaluate(self):
        """
        Evaluate the model on training and test datasets.
        
        Returns:
            Tuple[float, float]: Training and validation losses
        """
        self.model.eval()
        
        # Calculate training loss
        train_losses = []
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            loss = self._calculate_loss(inputs, targets)
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        
        # Calculate validation loss
        val_losses = []
        for inputs, targets in self.test_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            loss = self._calculate_loss(inputs, targets)
            val_losses.append(loss.item())
        validation_loss = np.mean(val_losses)
        
        return train_loss, validation_loss
    
    def train(self):
        """
        Public method to start the training process.
        
        Returns:
            The best model found during training
        """
        self._train()
        
        # Final evaluation after training
        final_train_loss, final_val_loss = self._evaluate()
        print(f"Final Train Loss: {final_train_loss:.4f}, Final Validation Loss: {final_val_loss:.4f}")
        
        # Load and return the best model
        if self.best_model_path:
            best_checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        return self.model
    
    def load_best_model(self, model=None):
        """
        Load the best saved model.
        
        Args:
            model (nn.Module, optional): Model to load state dict into. 
                                         If None, uses the original model.
        
        Returns:
            nn.Module: Model with best saved weights
        """
        if not self.best_model_path:
            raise ValueError("No best model has been saved yet.")
        
        if model is None:
            model = self.model
        
        checkpoint = torch.load(self.best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

if __name__ == '__main__':

    model = TamilGPT(vocab_size = 32000, embedding_dimension = 768, context_length = 256, num_heads = 12, scaling_factor = 4, num_layers = 12, bias = False, dropout = 0, weight_tying = True)
    trainer = Trainer(
        model=model, 
        train_loader=train_dataloader, 
        test_loader=val_dataloader, 
        loss=loss_function, 
        optimizer=optimizer, 
        epochs=5,
        checkpoint_dir='./checkpoints',
        model_name='tamilgpt'
    )

    # Train and get best model
    best_model = trainer.train()