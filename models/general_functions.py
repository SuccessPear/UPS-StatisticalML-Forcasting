import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

def train_step_regression(model: nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        optimizer: torch.optim.Optimizer,
                        loss_fn: torch.nn.Module,
                        device: torch.device = "cpu"):
    """_summary_

    Args:
        model (nn.Module): the ML model
        data_loader (torch.utils.data.DataLoader): the data loader that contrain training data
        optimizer (torch.optim.Optimizer): Optimizer to train the model
        loss_fn (torch.nn.Module): Loss function
        device (torch.device, optional): . Defaults to "cpu".

    Returns:
        Scalar: Train loss
    """
    model.train()
    
    total_loss = 0
    acc = 0
    for X, y in data_loader:
        #0. Move data to device
        X = X.to(device)
        y = y.to(device)
        
        #1. Forward pass
        y_pred = model(X)
        
        #2. Calculate loss and accuracy
        loss = loss_fn(y_pred, y)
        
        #3. Optimizer zero grad
        optimizer.zero_grad()
        
        #4. Loss backward (back propagation)
        loss.backward()
        
        #5. Optimizer step
        optimizer.step()
        
        # Add loss to total loss
        total_loss += loss
    # Average the loss by the length of dataloader
    total_loss /= len(data_loader)
    return total_loss

def test_step_regression(model: nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        loss_fn: torch.nn.Module,
                        device: torch.device = "cpu"):
    """_summary_

    Args:
        model (nn.Module): ML model
        data_loader (torch.utils.data.DataLoader): Dataloader that contain test  data
        loss_fn (torch.nn.Module): Loss function
        device (torch.device, optional):. Defaults to "cpu".

    Returns:
        Scalar: test loss
    """
    model.eval()
    with torch.inference_mode():
        total_loss = 0
        for X_test, y_test in data_loader:
            y_pred = model(X_test)
            test_loss = loss_fn(y_pred, y_test)
            total_loss += test_loss
            
        total_loss /= len(data_loader)
    return total_loss