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
    #global_loss = loss_fn(output, output) #--> to null the loss for experiments
    total_loss = alpha*recon_loss + (1 - alpha)*global_loss
    

    return recon_loss, global_loss, total_loss


def weight_update(model, optimizer, recon_loss, global_loss, total_loss):
    # Total loss for encoder
    # Recon loss for decoder
    # Global loss for classifier

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

    # Restore encoder and decoder gradients (from total and recon losses respectively)
    for param, grad in zip(model.encoder.parameters(), encoder_grads):
        param.grad = grad
        #param.grad = None # for frozen encoder experiment because trouble when module.freeze = 'encoder'
    for param, grad in zip(model.decoder.parameters(), decoder_grads):
        param.grad = grad

 


def print_grad_norms(model, name):
    for n, p in model.named_parameters():
        if p.grad is not None:
            print(f"{name} {n} grad norm: {p.grad.norm().item()}")
        else:
            print(f"{name} {n} grad: NONE")



def successive_learning(epoch, model):
    if epoch is not None:
        if (epoch <= 70): # unsupervised learning ONLY at the beginning of the learning
            model.freeze_module('classifier')
            return 1, 2 # alpha, lr
        elif epoch > 70: # unsupervised learning + classification after some epochs
            model.unfreeze_module('classifier')
            return 0.5, 0.05
    else:
        model.freeze_module('classifier')
        return 1, 2



def mix_error_signals(model, target, t, recon, thal_input, output, alpha):
    # Digits to mix (i.e exchange), their outputs after model pass & targets
    digit_x, digit_y = model.mix_signals
    mask_x = (target == digit_x)
    mask_y = (target == digit_y)
    num_mixs = min(mask_x.sum(), mask_y.sum())
    print("num_mixs: ", num_mixs)
    output_x = output[mask_x]
    output_y = output[mask_y]
    target_x = target[mask_x]
    target_y = target[mask_y] 
    t_x = F.one_hot(target_x, num_classes=10).float()
    t_y = F.one_hot(target_y, num_classes=10).float()

    # Compute loss of digits_x and digits_y seperately before mixing
    loss_fn = nn.MSELoss(reduction='none')
    loss_x = loss_fn(output_x, t_x)
    #print("len loss x:", len(loss_x), "loss_x: ", loss_x) OK
    loss_y = loss_fn(output_y, t_y)
    #print("len loss y:", len(loss_y),"loss_y: ", loss_y) OK
    mixed_loss_x = loss_y[:num_mixs]
    # check that mixed_loss_x is loss_y
    #print("len mixed_loss_x:", len(mixed_loss_x), "mixed_loss_x: ", mixed_loss_x) OK
    mixed_loss_y = loss_x[:num_mixs]
    # check that mixed_loss_y is loss_x
    #print("len mixed_loss_y:", len(mixed_loss_y), "mixed_loss_y: ", mixed_loss_y) OK
    final_loss_x = torch.cat([mixed_loss_x, loss_x[num_mixs:]])
    #print("len final_loss_x:", len(final_loss_x), "VS number of x", mask_x.sum()) OK
    final_loss_y = torch.cat([mixed_loss_y, loss_y[num_mixs:]])
    #print("len final_loss_y:", len(final_loss_y), " VS number of y", mask_y.sum()) OK 
  
    # Computing the rest of the global loss and inserting the mixed losses
    global_loss_without_mix = loss_fn(output, t)
    global_loss = global_loss_without_mix.clone()
    global_loss[mask_x] = final_loss_x
    global_loss[mask_y] = final_loss_y
    global_loss = global_loss.mean()
    global_loss_without_mix = global_loss_without_mix.mean()
    # print ("global_loss_without_mix : ", global_loss_without_mix, "VS global_loss : ", global_loss) 
    # print ("global loss without mix - global loss : ", global_loss_without_mix - global_loss) # seems like theres a diff

    # Compute the other losses normaly 
    recon_loss = loss_fn(recon, thal_input.view(thal_input.size(0), -1)).mean()
    total_loss = alpha*recon_loss + (1 - alpha)*global_loss

    return recon_loss, global_loss, total_loss



def swap_digits(model, target, recon, thal_input, output, loss_fn, alpha):
    # Digits to mix
    digit_x, digit_y = model.swap_digits
    mask_x = (target == digit_x)
    mask_y = (target == digit_y)

    # Swapping of digit X and digit Y
    target_swapped = target.clone()
    target_swapped[mask_x] = digit_y
    target_swapped[mask_y] = digit_x

    t_swapped = F.one_hot(target_swapped, num_classes=10).float()


    # Compute loss with swapped target (for classification task only because only supervised learning)
    recon_loss = loss_fn(recon, thal_input.view(thal_input.size(0), -1))
    global_loss = loss_fn(output, t_swapped)
    total_loss = alpha*recon_loss + (1 - alpha)*global_loss
    
    return recon_loss, global_loss, total_loss



def compute_scale_factor(current_epoch, swap_epoch, min_scale=0.05, decay_rate=0.95):
    epochs_since_swap = current_epoch - swap_epoch
    return max(min_scale, decay_rate ** epochs_since_swap)



def scaled_backward(model, scale_factor):
    for param in model.parameters():
        if param.grad is not None:
            param.grad *= scale_factor



def weight_update_v1(model, optimizer, recon_loss, global_loss, total_loss):
    # Total loss for encoder
    # Recon loss for decoder
    # Global loss for classifier
    # For forzen encoder

    # Updating encoder weights with total_loss (combination weighted of recon_loss and global_loss)
    if model.encoder.parameters().__next__().requires_grad:
        total_loss.backward(retain_graph=True)
        encoder_grads = [param.grad.clone() if param.grad is not None else None 
                    for param in model.encoder.parameters()]
        optimizer.zero_grad()
    else : 
        encoder_grads = None

    # Updating decoder weights with recon_loss
    if model.decoder.parameters().__next__().requires_grad:
        recon_loss.backward(retain_graph=True)
        decoder_grads = [param.grad.clone() if param.grad is not None else None 
                    for param in model.decoder.parameters()]
        optimizer.zero_grad()
    else:
        decoder_grads = None
        

    # Updating classifier weights with global_loss
    if model.classifier.parameters().__next__().requires_grad:
        if global_loss.requires_grad:
            global_loss.backward(retain_graph=True)
            classifier_grads = [param.grad.clone() if param.grad is not None else None
                    for param in model.classifier.parameters()]
            optimizer.zero_grad()
        else : 
            classifier_grads = None
    else:
        classifier_grads = None


    # Restore encoder and decoder gradients (from total and recon losses respectively)
    for param, grad in zip(model.encoder.parameters(), encoder_grads):
        param.grad = grad
    for param, grad in zip(model.decoder.parameters(), decoder_grads):
        param.grad = grad
    for param, grad in zip(model.classifier.parameters(), classifier_grads):
        param.grad = grad