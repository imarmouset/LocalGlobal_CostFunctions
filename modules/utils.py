

def calculate_ssim(recon, original):
    pass
    recon = recon.detach().cpu().numpy().reshape(-1, 28, 28)
    original = original.detach().cpu().numpy().reshape(-1, 28, 28)
    #ssim_scores = [ssim(r, o, data_range=1) for r, o in zip(recon, original)]
    #return np.mean(ssim_scores)



def weight_update(model, optimizer, loss, recon_loss):
    optimizer.zero_grad()
    non_decoder_grads = {}

    loss.backward(retain_graph=True)
    for name, param in model.named_parameters():
        if 'Decoder' not in name:
            non_decoder_grads[name] = param.grad.clone()
        param.grad = None
    
    recon_loss.backward()
    for name, param in model.named_parameters():
        if 'Decoder' not in name:
            param.grad = non_decoder_grads[name]
    
    optimizer.step()

def loss_local(loss_fn, PV_pred, Pyr_out, recon, inputPyr, pred_coeff):
    inputPyr = inputPyr.view(-1, 784)
    pred_loss = loss_fn(Pyr_out, PV_pred)
    recon_loss = loss_fn(recon, inputPyr)
    loss = pred_coeff * pred_loss + recon_loss
    return loss, pred_loss, recon_loss


def loss_combined(loss_fn, t, PV_pred, Pyr_pred, Pyr_out, recon, inputPyr):
    pred_coeff = 1
    inputPyr = inputPyr.view(-1, 784)
    pred_loss = loss_fn(Pyr_pred, PV_pred)
    recon_loss = loss_fn(recon, inputPyr)
    SSLloss = pred_coeff * pred_loss + recon_loss
    global_loss = loss_fn(Pyr_out, t)
    total_loss = global_loss + SSLloss
    return total_loss, global_loss, pred_loss, recon_loss

