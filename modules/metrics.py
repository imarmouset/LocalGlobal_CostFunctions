import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim



def calculate_accuracy(targets, outputs, total, correct):
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        return total, correct, 100 * correct / total




def visualize_tsne(encoded_outputs, labels, epoch=None):
    n_samples = 1000  # otherwise takes too much time to use all the points
    indices = np.random.choice(encoded_outputs.shape[0], n_samples, replace=False)
    encoded_outputs = encoded_outputs[indices]
    labels = labels[indices]
    tsne = TSNE(n_components=2, random_state=42)
    encoded_outputs_2d = tsne.fit_transform(encoded_outputs)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(encoded_outputs_2d[:, 0], encoded_outputs_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    
    if epoch is not None:
        plt.title(f't-SNE visualization of encoded outputs - Epoch {epoch}')
        plt.savefig(f'./Plots/tSNE/tsne_epoch_{epoch}.png')
    else:
        plt.title('t-SNE visualization of encoded outputs - Epoch 0')
        plt.savefig('./Plots/tSNE/tsne_epoch_0.png')   
    plt.close()




def acc_swapped(output, target, swap_argument):
    digit_x, digit_y = swap_argument
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == target).float()
    
    mask_x = (target == digit_x)
    mask_y = (target == digit_y)
    mask_other = ~(mask_x | mask_y)
    
    accuracy_x = correct[mask_x].mean().item() if mask_x.any() else 0
    accuracy_y = correct[mask_y].mean().item() if mask_y.any() else 0
    accuracy_other = correct[mask_other].mean().item() if mask_other.any() else 0
    
    return (accuracy_x, accuracy_y, accuracy_other)




def plot_digits(model, test_loader, n=5, swap_argument=None, epoch=None):

    plt.figure(figsize=(20, 8))
    data, labels = next(iter(test_loader))
    
    if swap_argument is not None:
        digit_x, digit_y = swap_argument
        idx_x = (labels == digit_x).nonzero().flatten()
        idx_y = (labels == digit_y).nonzero().flatten()
        idx_other = ((labels != digit_x) & (labels != digit_y)).nonzero().flatten()
        
        selected_idx = np.concatenate([
            np.random.choice(idx_x, 2, replace=False),
            np.random.choice(idx_y, 2, replace=False),
            np.random.choice(idx_other, 1, replace=False)
        ])
    else:
        selected_idx = np.random.choice(len(labels), n, replace=False)
    
    selected_data = data[selected_idx]
    selected_labels = labels[selected_idx]
    model.eval()
    with torch.no_grad():
        _, recon, _ = model(selected_data)
    
    for i in range(n):
        # Original 
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(selected_data[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"Original: {selected_labels[i].item()}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)        
        # Reconstructed 
        ax = plt.subplot(2, n, i + 1 + n)
        recon_img = recon[i].squeeze().cpu().numpy().reshape(28, 28)
        plt.imshow(recon_img, cmap='gray')
        plt.title("Reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    if epoch is not None:
        plt.suptitle(f'Digits visualization - Epoch {epoch}')
        plt.savefig(f'./Plots/Digits/digits_epoch_{epoch}.png')
    else:
        plt.suptitle('Digits visualization - Epoch 0')
        plt.savefig('./Plots/Digits/digits_epoch_0.png')   
    plt.close()


def calculate_ssim(recon, original):
    recon = recon.detach().cpu().numpy().reshape(-1, 28, 28)
    original = original.detach().cpu().numpy().reshape(-1, 28, 28)
    ssim_scores = [ssim(r, o, data_range=1) for r, o in zip(recon, original)]
    return np.mean(ssim_scores)


class AccuracyTracker:
    def __init__(self, swap_argument):
        self.digit_x, self.digit_y = swap_argument
        self.correct_x = 0
        self.correct_y = 0
        self.correct_other = 0
        self.total_x = 0
        self.total_y = 0
        self.total_other = 0

    def update(self, output, target):
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == target)
        
        mask_x = (target == self.digit_x)
        mask_y = (target == self.digit_y)
        mask_other = ~(mask_x | mask_y)

        self.correct_x += correct[mask_x].sum().item()
        self.correct_y += correct[mask_y].sum().item()
        self.correct_other += correct[mask_other].sum().item()

        self.total_x += mask_x.sum().item()
        self.total_y += mask_y.sum().item()
        self.total_other += mask_other.sum().item()

    def get_accuracies(self):
        acc_x = self.correct_x / self.total_x if self.total_x > 0 else 0
        acc_y = self.correct_y / self.total_y if self.total_y > 0 else 0
        acc_other = self.correct_other / self.total_other if self.total_other > 0 else 0
        return (acc_x, acc_y, acc_other)


