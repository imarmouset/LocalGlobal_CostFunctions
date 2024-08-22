import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import silhouette_score


def calculate_ssim(recon, original):
    recon = recon.detach().cpu().numpy().reshape(-1, 28, 28)
    original = original.detach().cpu().numpy().reshape(-1, 28, 28)
    ssim_scores = [ssim(r, o, data_range=1) for r, o in zip(recon, original)]
    return np.mean(ssim_scores)




def calculate_accuracy(targets, output, total, correct):
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        return total, correct, 100 * correct / total



def calculate_multiple_accuracy(targets, output, total, correct):
    _, predicted = output.max(1)
    # Iterate through the targets and predictions
    for digit in range(10):
        # Create a mask for the current digit
        mask = (targets == digit)
        # Update total count for the current digit
        total[digit] += mask.sum().item()
        # Update correct count for the current digit
        correct[digit] += (predicted[mask] == targets[mask]).sum().item()
    # don't forget to compute the actual accuracy at the end of the epoch! 






def plot_digits_and_saliency(model, test_loader, n=5, swap_argument=None, epoch=None):
    fig = plt.figure(figsize=(20, 15))  
    data, labels = next(iter(test_loader))
    
    if swap_argument is not None:
        digit_x, digit_y = swap_argument
        idx_x = (labels == digit_x).nonzero().flatten()
        idx_y = (labels == digit_y).nonzero().flatten()
        idx_other = ((labels != digit_x) & (labels != digit_y)).nonzero().flatten()
        selected_idx = np.concatenate([
            np.random.choice(idx_x, 2, replace=False),
            np.random.choice(idx_y, 2, replace=False),
            np.random.choice(idx_other, 2, replace=False)
        ])
    else:
        selected_idx = np.random.choice(len(labels), n, replace=False)
    
    selected_data = data[selected_idx]
    selected_labels = labels[selected_idx]
    
    model.eval()
    
    for i in range(n):
        image = selected_data[i].unsqueeze(0)
        label = selected_labels[i]
        
        # Computation of saliency map
        image.requires_grad_()
        *_, _, output = model(image)
        output[0, label].backward()
        saliency, _ = torch.max(image.grad.data.abs(), dim=1)
        saliency = saliency.reshape(28, 28)
        
        # Original image
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(image.squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title(f"Original: {label.item()}", fontsize=12, pad=5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Reconstructed: from the auto-encoder
        with torch.no_grad():
            *_, recon, _ = model(image)
        ax = plt.subplot(3, n, i + 1 + n)
        recon_img = recon.squeeze().cpu().numpy().reshape(28, 28)
        plt.imshow(recon_img, cmap='gray')
        plt.title("Reconstructed", fontsize=12, pad=5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Saliency map: classifier metric
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(saliency.detach().cpu().numpy(), cmap='hot')
        plt.colorbar()
        plt.title("Saliency map", fontsize=12, pad=5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    if epoch is not None:
        plt.suptitle(f'Digits visualization: Original vs Reconstructed vs Saliency map - Epoch {epoch}', fontsize=16, y=0.98)
        plt.savefig(f'./Plots/Digits/digits_saliency_epoch_{epoch}.png')
    else:
        plt.suptitle('Digits visualization: Original vs Reconstructed vs Saliency map - Epoch 0', fontsize=16, y=0.98)
        plt.savefig('./Plots/Digits/digits_saliency_epoch_0.png')
    
    plt.close()



def plot_digits(model, test_loader, n=5, swap_argument=None, epoch=None):
    fig = plt.figure(figsize=(20, 10))
    data, labels = next(iter(test_loader))
    if swap_argument is not None:
        digit_x, digit_y = swap_argument
        idx_x = (labels == digit_x).nonzero().flatten()
        idx_y = (labels == digit_y).nonzero().flatten()
        idx_other = ((labels != digit_x) & (labels != digit_y)).nonzero().flatten()
        selected_idx = np.concatenate([
        np.random.choice(idx_x, 2, replace=False),
        np.random.choice(idx_y, 2, replace=False),
        np.random.choice(idx_other, 2, replace=False)
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
        plt.title(f"Original: {selected_labels[i].item()}", fontsize=12, pad=5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        recon_img = recon[i].squeeze().cpu().numpy().reshape(28, 28)
        plt.imshow(recon_img, cmap='gray')
        plt.title("Reconstructed", fontsize=12, pad=5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if epoch is not None:
        plt.suptitle(f'Digits visualization: Original vs reconstructed by the auto-encoder - Epoch {epoch}', fontsize=16, y=0.98)
        plt.savefig(f'./Plots/Digits/digitsepoch{epoch}.png')
    else:
        plt.suptitle('Digits visualization: Original vs reconstructed by the auto-encoder - Epoch 0', fontsize=16, y=0.98)
        plt.savefig('./Plots/Digits/digits_epoch_0.png')
    plt.close()




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





def tsne_and_clustering(encoded_outputs, labels, epoch=None):
    n_samples = 1000  # otherwise takes too much time to use all the points
    indices = np.random.choice(encoded_outputs.shape[0], n_samples, replace=False)
    encoded_outputs = encoded_outputs[indices]
    labels = labels[indices]

    # t-SNE & silhouette score computation
    tsne = TSNE(n_components=2, random_state=42)
    encoded_outputs_2d = tsne.fit_transform(encoded_outputs)  
    silhouette = silhouette_score(encoded_outputs, labels)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(encoded_outputs_2d[:, 0], encoded_outputs_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    
    if epoch is not None:
        plt.title(f't-SNE visualization of encoded outputs - Epoch {epoch}\nSilhouette Score: {silhouette:.3f}')
        plt.savefig(f'./Plots/tSNE/tsne_epoch_{epoch}.png')
    else:
        plt.title(f't-SNE visualization of encoded outputs - Epoch 0\nSilhouette Score: {silhouette:.3f}')
        plt.savefig('./Plots/tSNE/tsne_epoch_0.png')
    
    plt.close()
    
    return silhouette
