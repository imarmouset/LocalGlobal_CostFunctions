

def calculate_ssim(recon, original):
    pass
    recon = recon.detach().cpu().numpy().reshape(-1, 28, 28)
    original = original.detach().cpu().numpy().reshape(-1, 28, 28)
    #ssim_scores = [ssim(r, o, data_range=1) for r, o in zip(recon, original)]
    #return np.mean(ssim_scores)



def weight_update(model, optimizer, loss, recon_loss):
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    non_decoder_grads = {}

    for name, param in model.named_parameters():
        if 'Decoder' not in name:
            non_decoder_grads[name] = param.grad.clone()
        param.grad = None
    
    recon_loss.backward()
    for name, param in model.named_parameters():
        if 'Decoder' not in name:
            param.grad = non_decoder_grads[name]
    
    optimizer.step()

def losses(loss_fn, PV_out, PV_pred, Pyr_out, recon, inputs1, inputs2, pred_coeff):
    inputs2 = inputs2.view(-1, 784)
    pred_loss = loss_fn(Pyr_out, PV_pred)
    recon_loss = loss_fn(recon, inputs2)
    loss = pred_coeff * pred_loss + recon_loss
    return loss, pred_loss, recon_loss

