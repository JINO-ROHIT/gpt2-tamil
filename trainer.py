import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numpy as np
import multiprocessing
import time

from layers import TamilGPT
from utils import create_lazy_split_dataloader

class VerboseTrainer:
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
        Initialize the Trainer with enhanced verbosity and detailed logging.
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
        
        # Tracking additional metrics
        self.training_metrics = {
            'total_steps': 0,
            'epoch_losses': [],
            'validation_losses': [],
            'training_times': []
        }
        
        # Print out initial model and training configuration
        self._print_config()
    
    def _print_config(self):
        """
        Print out detailed configuration of the training setup.
        """
        print("\n===== Training Configuration =====")
        print(f"Model: {type(self.model).__name__}")
        print(f"Device: {self.device}")
        print(f"Total Epochs: {self.epochs}")
        print(f"Batch Size: {self.train_loader.batch_size}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Loss Function: {type(self.loss).__name__}")
        print("==================================\n")
    
    def _calculate_loss(self, inputs, targets):
        """
        Calculate loss with more detailed logging.
        """
        logits = self.model(inputs)
        loss = self.loss(logits.flatten(0, 1), targets.flatten())
        
        # Optional: Add gradient norm logging
        if self.training_metrics['total_steps'] % 50 == 0:
            total_grad_norm = self._compute_gradient_norm()
            print(f"Step {self.training_metrics['total_steps']} - Gradient Norm: {total_grad_norm:.4f}")
        
        return loss
    
    def _compute_gradient_norm(self):
        """
        Compute the total gradient norm across all parameters.
        """
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """
        Enhanced checkpoint saving with more verbose logging.
        """
        if val_loss < self.best_val_loss:
            print("\n🏆 New Best Model Found! 🏆")
            print(f"Validation Loss Improved: {self.best_val_loss:.4f} → {val_loss:.4f}")
            
            self.best_val_loss = val_loss
            
            checkpoint_filename = f"{self.model_name}_epoch{epoch+1}_loss{val_loss:.4f}.pth"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_loss,
                'training_metrics': self.training_metrics
            }, checkpoint_path)
            
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
            
            self.best_model_path = checkpoint_path
    
    def _train(self) -> None:
        """
        Train the model with extensive logging and progress tracking.
        """
        print("\n🚀 Starting Model Training 🚀")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            print(f"\n--- Epoch {epoch+1}/{self.epochs} ---")
            
            self.model.train()
            epoch_loss = 0.0
            batch_losses = []
            
            progress_bar = tqdm(enumerate(self.train_loader), 
                                #total=len(self.train_loader), 
                                desc=f"Epoch {epoch+1}")
            
            for batch_idx, (inputs, targets) in progress_bar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                loss = self._calculate_loss(inputs, targets)
                loss.backward()
                self.optimizer.step()
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_losses.append(batch_loss)
                
                self.training_metrics['total_steps'] += 1
                
                # Update progress bar with current metrics
                progress_bar.set_postfix({
                    'Batch Loss': f'{batch_loss:.4f}', 
                    'Avg Batch Loss': f'{np.mean(batch_losses):.4f}'
                })
            
            # Epoch summary
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            self.training_metrics['training_times'].append(epoch_duration)
            
            # Evaluate and log epoch results
            train_loss, validation_loss = self._evaluate()
            self.training_metrics['epoch_losses'].append(train_loss)
            self.training_metrics['validation_losses'].append(validation_loss)
            
            print(f"\n📊 Epoch Summary:")
            print(f"Total Epoch Loss: {epoch_loss:.4f}")
            print(f"Average Batch Loss: {epoch_loss/len(self.train_loader):.4f}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {validation_loss:.4f}")
            print(f"Epoch Duration: {epoch_duration:.2f} seconds")
            
            # Save checkpoint
            self._save_checkpoint(epoch, validation_loss)
        
        total_training_time = time.time() - start_time
        print(f"\n🏁 Training Complete! Total Time: {total_training_time:.2f} seconds")
    
    @torch.inference_mode()
    def _evaluate(self):
        """
        Evaluate the model with more comprehensive metrics.
        """
        self.model.eval()
        
        # Detailed evaluation metrics
        train_losses, val_losses = [], []
        
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            loss = self._calculate_loss(inputs, targets)
            train_losses.append(loss.item())
        
        for inputs, targets in self.test_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            loss = self._calculate_loss(inputs, targets)
            val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        validation_loss = np.mean(val_losses)
        
        return train_loss, validation_loss
    
    def train(self):
        """
        Public method to start the training process with comprehensive tracking.
        
        Returns:
            The best model found during training
        """
        self._train()
        
        # Final evaluation and model loading
        final_train_loss, final_val_loss = self._evaluate()
        print(f"\n🔍 Final Evaluation:")
        print(f"Final Train Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        
        # Detailed training metrics report
        self._print_training_report()
        
        # Load and return the best model
        if self.best_model_path:
            best_checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        return self.model
    
    def _print_training_report(self):
        """
        Print a comprehensive report of training metrics.
        """
        print("\n📈 Training Metrics Report 📈")
        print(f"Total Training Steps: {self.training_metrics['total_steps']}")
        
        if self.training_metrics['epoch_losses']:
            print(f"Average Training Loss: {np.mean(self.training_metrics['epoch_losses']):.4f}")
            print(f"Best Training Loss: {min(self.training_metrics['epoch_losses']):.4f}")
        
        if self.training_metrics['validation_losses']:
            print(f"Average Validation Loss: {np.mean(self.training_metrics['validation_losses']):.4f}")
            print(f"Best Validation Loss: {min(self.training_metrics['validation_losses']):.4f}")
        
        if self.training_metrics['training_times']:
            print(f"Average Epoch Duration: {np.mean(self.training_metrics['training_times']):.2f} seconds")
            print(f"Total Training Time: {sum(self.training_metrics['training_times']):.2f} seconds")

if __name__ == '__main__':
    model = TamilGPT(
        vocab_size=32000, 
        embedding_dimension=768, 
        context_length=256, 
        num_heads=12, 
        scaling_factor=4, 
        num_layers=12, 
        bias=False, 
        dropout=0, 
        weight_tying=True
    )

    train_dataloader, val_dataloader = create_lazy_split_dataloader(
        file_path='data/sample.txt',
        batch_size=32,
        max_length=256,
        stride=1,
        train_ratio=0.8,
        num_workers=multiprocessing.cpu_count() // 2,
        seed=42
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    trainer = VerboseTrainer(
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