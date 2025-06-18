import torch
import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot training and testing metrics"""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), facecolor='black')
    fig.patch.set_facecolor('black')
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', color='cyan')
    ax1.plot(history['test_loss'], label='Test Loss', color='magenta')
    ax1.set_title('Loss History', color='white', pad=20)
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('Loss', color='white')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Acc', color='cyan')
    ax2.plot(history['test_acc'], label='Test Acc', color='magenta')
    ax2.set_title('Accuracy History', color='white', pad=20)
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Accuracy (%)', color='white')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Load the saved model and history
checkpoint = torch.load('final_nam_model.pt')
history = checkpoint['history']

# Plot the results
plot_training_history(history)

# Print best test accuracy achieved
print(f"Best test accuracy: {checkpoint['best_test_acc']:.2f}%") 