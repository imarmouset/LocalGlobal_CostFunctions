

def calculate_ssim(recon, original):
    pass
    recon = recon.detach().cpu().numpy().reshape(-1, 28, 28)
    original = original.detach().cpu().numpy().reshape(-1, 28, 28)
    #ssim_scores = [ssim(r, o, data_range=1) for r, o in zip(recon, original)]
    #return np.mean(ssim_scores)



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

