import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.aux import JS_divergence

def distill_knowledge(opt, epoch, student, teacher, mem_generator, optimizer_S, gen_imgs):
    optimizer_S.zero_grad()

    # Infer memory Samples
    if ((epoch % opt.student_rehearse_period) == 0):
        mem_generator.eval()
        student.eval()
        z_mem = torch.randn((opt.num_mem_samples, opt.latent_dim), requires_grad=True, device='cuda')
        optimizer_Z = torch.optim.Adam([z_mem], lr=opt.lr_G)
        optimizer_Z.zero_grad()
        for z_epoch in range(1):
            log_var = torch.var(z_mem, dim=0)
            mu = torch.mean(z_mem, dim=0)
            gen_imgs_mem = mem_generator(z_mem)
            outputs_T = teacher(gen_imgs_mem)
            softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
            loss_one_hot = torch.nn.CrossEntropyLoss()(outputs_T,outputs_T.data.max(1)[1])
            loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()

            loss_kld = F.mse_loss(log_var, torch.ones(opt.latent_dim).cuda()) + F.mse_loss(mu, torch.zeros(opt.latent_dim).cuda())
            loss_match = loss_information_entropy*opt.ie + loss_one_hot*(opt.oh)
            (loss_match + loss_kld).backward()
            optimizer_Z.step()

        gen_imgs_mem = mem_generator(z_mem).detach()
        gen_imgs = torch.cat((gen_imgs, gen_imgs_mem),dim=0)
        student.train()

    outputs_S = student(gen_imgs)
    outputs_T = teacher(gen_imgs)
    
    loss_kd = F.l1_loss(outputs_S, outputs_T.detach())
    loss_kd.backward()
    optimizer_S.step()

    return loss_kd

def train_novel_generator(opt, generator, optimizer_G, teacher, student):
    optimizer_G.zero_grad()

    # Sample novel synthetic data
    z = Variable(torch.randn(opt.batch_size, opt.latent_dim), requires_grad=False).cuda()
    gen_imgs = generator(z)
     
    # Get teacher responses to match training data dist. 
    outputs_T, features_T = teacher(gen_imgs, out_feature=True)     
    pred = outputs_T.data.max(1)[1]
    loss_activation = -features_T[-1].abs().mean()
    loss_one_hot = torch.nn.CrossEntropyLoss()(outputs_T,pred)
    softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
    loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
    loss_match_T = loss_one_hot * opt.oh + loss_information_entropy * opt.ie + loss_activation * opt.a

    # Calculate T-S disagreement
    loss_disaggreement = JS_divergence(student(gen_imgs), outputs_T.detach())

    loss_gen = loss_match_T + loss_disaggreement
    losses = [loss_gen, loss_disaggreement]

    loss_gen.backward()
    optimizer_G.step()

    return losses

def train_memory_generator(opt, epoch, i, mem_generator, encoder, optimizer_G2, optimizer_E, teacher, gen_imgs):
    mem_generator.train()
    encoder.train()
    optimizer_G2.zero_grad()
    optimizer_E.zero_grad()
    
    # If memory generator has not seen any samples just use novel samples
    if (i == 0) & (epoch == 0):
        outputs_T = teacher(gen_imgs)
        z_enc, mu, log_var = encoder(gen_imgs[:(opt.batch_size // 2)].detach())
        gen_imgs_rec = mem_generator(z_enc)
        out_T_target = [gen_imgs[:(opt.batch_size // 2)].detach()]
        out_T_target.extend([outputs_T[:(opt.batch_size // 2)]])
        #out_T_target.extend(features_T[-1:])

        rec_pred_T = teacher(gen_imgs_rec)
        rec_out_T = [gen_imgs_rec]
        rec_out_T.extend([rec_pred_T])
        #rec_out_T.extend(rec_features_T[-1:])
        
        # Reconstruction Loss
        recons_loss = 0.
        for rec_idx, rec_target in enumerate(out_T_target):
            recons_loss += F.l1_loss(rec_out_T[rec_idx], rec_target[:(opt.batch_size // 2)].detach())

    else:
        z_mem = torch.randn((opt.num_mem_samples, opt.latent_dim), requires_grad=True, device='cuda')
        optimizer_Z = torch.optim.Adam([z_mem], lr=opt.lr_G)
        optimizer_Z.zero_grad()
        
        # Tune latent variables
        for z_epoch in range(1):
            log_var = torch.var(z_mem, dim=0)
            mu = torch.mean(z_mem, dim=0)
            gen_imgs_mem = mem_generator(z_mem)
            outputs_T = teacher(gen_imgs_mem)
            softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
            loss_one_hot = torch.nn.CrossEntropyLoss()(outputs_T,outputs_T.data.max(1)[1])
            loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
            loss_kld = F.mse_loss(log_var, torch.ones(opt.latent_dim).cuda()) + F.mse_loss(mu, torch.zeros(opt.latent_dim).cuda())

            loss_match = loss_information_entropy*opt.ie + loss_one_hot*(opt.oh)
            (loss_match + loss_kld).backward()
            optimizer_Z.step()

        # Self infer memory samples
        gen_imgs_mem = mem_generator(z_mem).detach()
        concat_imgs = torch.cat((gen_imgs[:(opt.batch_size // 2)].detach(), gen_imgs_mem),dim=0)
        pred_T_target = teacher(concat_imgs)

        out_T_target = [concat_imgs]
        out_T_target.extend([pred_T_target])
        #out_T_target.extend(features_T[-1:])
        
        z_enc, mu, log_var = encoder(concat_imgs)
        gen_imgs_rec = mem_generator(z_enc)
        rec_pred_T = teacher(gen_imgs_rec)
        rec_out_T = [gen_imgs_rec]
        rec_out_T.extend([rec_pred_T])
        #rec_out_T.extend(rec_features_T[-1:])

        # Reconstruction Loss
        recons_loss = 0.
        for rec_idx, rec_target in enumerate(out_T_target):
            recons_loss += F.l1_loss(rec_out_T[rec_idx], rec_target.detach())

    # KLD Loss
    kld_loss = 1e-5 * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    mem_gen_loss = recons_loss + kld_loss
    
    mem_gen_loss.backward()

    optimizer_G2.step()
    optimizer_E.step()

    return mem_gen_loss