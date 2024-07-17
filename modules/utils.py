from skimage.metrics import structural_similarity as ssim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn




def compute_losses(recon, thal_input, output, t, loss_fn, alpha):
    recon_loss = loss_fn(recon, thal_input.view(thal_input.size(0), -1)) 
    global_loss = loss_fn(output, t)
    total_loss = alpha*recon_loss + (1 - alpha)*global_loss
    #total_loss = loss_fn(output, output)

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


def mix_error_signals(model, target, t, recon, thal_input, output, alpha):

    # Which digits of the batch are concerned? How many digits? 
    digit_x, digit_y = model.swap_digits
    mask_x = (target == digit_x)
    mask_y = (target == digit_y)
    num_mixs = min(mask_x.sum(), mask_y.sum())
    print(f"Number of digit {digit_x}: {mask_x.sum()}; number of digit {digit_y}: {mask_y.sum()}; should be {num_mixs} exchanges")

    # Selecting the outputs & their target for digit_x and digit_y
    output_x = output[mask_x]
    output_y = output[mask_y]
    target_x = t[mask_x]
    target_y = t[mask_y] 

    # Compute loss of digits_x and digits_y seperately before mixing
    loss_fn = nn.MSELoss(reduction='none')
    loss_x = loss_fn(output_x, target_x)
    loss_y = loss_fn(output_y, target_y)

    # Mixing the losses for num_mixs units, and concatenate with the losses that have not been mixed
    swapped_loss_x = loss_y[:num_mixs]
    swapped_loss_y = loss_x[:num_mixs]
    final_loss_x = torch.cat([swapped_loss_x, loss_x[num_mixs:]])
    final_loss_y = torch.cat([swapped_loss_y, loss_y[num_mixs:]])
  

    # Computing the rest of the global loss and inserting the mixed losses
    no_mix_global_loss = loss_fn(output, t)
    print(f"No mix Global loss shape: {no_mix_global_loss.shape}")
    print(f"No mix Global loss: {no_mix_global_loss}")
    global_loss = loss_fn(output, t)
    print(f"Global loss shape: {global_loss.shape}")
    global_loss[mask_x] = final_loss_x
    global_loss[mask_y] = final_loss_y
    print(f"Global loss: {global_loss}")

    # Check the differences between global_loss and no_mix_global_loss
 
    print(f"{(global_loss != no_mix_global_loss).sum()} differences between global_loss and no_mix_global_loss")

    # Compute the other losses normaly
    recon_loss = loss_fn(recon, thal_input.view(thal_input.size(0), -1))
    total_loss = alpha*recon_loss + (1 - alpha)*global_loss

    return recon_loss, global_loss, total_loss



def swap_digits(model, target, t, recon, thal_input, output, loss_fn, alpha):
    # Which digits of tha batch are concerned? How many digits?
    digit_x, digit_y = model.swap_digits
    mask_x = (target == digit_x)
    mask_y = (target == digit_y)
    num_x = mask_x.sum()
    num_y = mask_y.sum()
    print(f"Number of digit {digit_x}: {num_x}; number of digit {digit_y}: {num_y}; should be {num_x + num_y} modifications")

    # Modify the label of digit X by digit Y and vice versa
    t_swapped = t.clone()
    t_swapped[mask_x] = digit_y
    t_swapped[mask_y] = digit_x

    print(f"Number of differences t vs t_swapped: {(t != t_swapped).sum()}. Equal to theoretical {num_x + num_y}? ")


    # Compute loss with swapped target for classification task ONLY
    recon_loss, _, _ = compute_losses(recon, thal_input, output, t, loss_fn, alpha)
    _, global_loss, _ = compute_losses(recon, thal_input, output, t_swapped, loss_fn, alpha)
    total_loss = alpha*recon_loss + (1 - alpha)*global_loss

    #print(f"Original global loss: {compute_losses(recon, thal_input, output, t, loss_fn, alpha)[1]}")
    #print(f"Swapped global loss: {global_loss}")    

    return recon_loss, global_loss, total_loss