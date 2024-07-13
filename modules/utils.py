from skimage.metrics import structural_similarity as ssim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



def compute_losses(recon, thal_input, output, t, loss_fn, alpha):
    recon_loss = loss_fn(recon, thal_input.view(thal_input.size(0), -1)) 
    global_loss = loss_fn(output, t)
    #total_loss = alpha*recon_loss + (1 - alpha)*global_loss
    total_loss = loss_fn(output, output)

    return recon_loss, global_loss, total_loss


def weight_update(model, optimizer, recon_loss, global_loss, total_loss):

    # Updating encoder weights with total_loss (combination weighted of recon_loss and global_loss)
    total_loss.backward(retain_graph=True)
    encoder_grads = [param.grad.clone() if param.grad is not None else None 
                 for param in model.encoder.parameters()]
    optimizer.zero_grad()

    # Updating decoder weights with recon_loss
    recon_loss.backward(retain_graph=True)
    decoder_grads = [param.grad.clone() if param.grad is not None else None 
                 for param in model.decoder.parameters()]
    optimizer.zero_grad()

    # Updating classifier weights with global_loss
    global_loss.backward()

    # Restore encoder and decoder gradient (from total and recon losses respectively)
    for param, grad in zip(model.encoder.parameters(), encoder_grads):
        param.grad = grad
    for param, grad in zip(model.decoder.parameters(), decoder_grads):
        param.grad = grad

    optimizer.step()



def print_grad_norms(model, name):
    for n, p in model.named_parameters():
        if p.grad is not None:
            print(f"{name} {n} grad norm: {p.grad.norm().item()}")
        else:
            print(f"{name} {n} grad: NONE")


def calculate_ssim(recon, original):
    recon = recon.detach().cpu().numpy().reshape(-1, 28, 28)
    original = original.detach().cpu().numpy().reshape(-1, 28, 28)
    ssim_scores = [ssim(r, o, data_range=1) for r, o in zip(recon, original)]
    return np.mean(ssim_scores)


def calculate_accuracy(targets, outputs, total, correct):
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        return total, correct, 100 * correct / total


def visualize_tsne(encoded_outputs, labels, epoch=None):
    n_samples = 1000  # or however many you want
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