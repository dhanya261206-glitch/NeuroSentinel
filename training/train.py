import torch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
import time

# 1. Post-processing tools
# Converts AI output (probabilities) into clear 0 or 1 masks
post_trans = AsDiscrete(threshold=0.5)
dice_metric = DiceMetric(include_background=True, reduction="mean")

# 2. Training Parameters
num_epochs = 20
val_interval = 2 # Check accuracy every 2 epochs
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

print("Starting Training...")

for epoch in range(num_epochs):
    print("-" * 10)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        optimizer.zero_grad() # Reset the "gradients" (slopes)
        outputs = model(inputs) # AI makes a guess
        
        # Calculate Loss
        loss = loss_function(outputs, labels)
        
        loss.backward() # Calculate how to fix weights
        optimizer.step() # Update the weights
        
        epoch_loss += loss.item()
        if step % 20 == 0:
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
            
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"Average Loss: {epoch_loss:.4f}")

    # 3. Validation (Checking how good the AI is)
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                
                # Sliding window is used because full images are too big
                val_outputs = sliding_window_inference(val_inputs, (128, 128, 128), 4, model)
                
                # Convert to 0 or 1
                val_outputs = [post_trans(i) for i in val_outputs]
                
                # Compute Dice Score
                dice_metric(y_pred=val_outputs, y=val_labels)
            
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)
            
            # Save the BEST model
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model.pth")
                print("Saved new best model!")
            
            print(f"Current Epoch: {epoch + 1}, Current Dice: {metric:.4f}, Best Dice: {best_metric:.4f}")

print(f"Training Complete! Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")