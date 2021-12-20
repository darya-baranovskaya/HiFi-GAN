from torch import nn
import torch
import wandb
from torch.nn import functional as F
from utils.melspectrogram import *

device = torch.device('cuda')

def model_require_grad(model, require_grad):
    for param in model.parameters():
        param.require_grad = require_grad

class L1MelLoss():
    def __init__(self):
        self.featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
        self.loss_fn = nn.L1Loss()

    def __call__(self, pred_wav, inp):
        out_mel = self.featurizer(pred_wav)
        if out_mel.shape[-1] < inp.shape[-1]:
            out_mel = torch.cat(
                [out_mel, -11.5129 * torch.ones((out_mel.shape[0], 80, inp.shape[-1] - out_mel.shape[-1])).to(device)],
                dim=-1)
        elif out_mel.shape[-1] > inp.shape[-1]:
            with torch.no_grad():
                inp = torch.cat(
                    [inp, -11.5129 * torch.ones((inp.shape[0], 80, out_mel.shape[-1] - inp.shape[-1])).to(device)],
                    dim=-1)
        return self.loss_fn(out_mel, inp)


class GANTraining():
    def __init__(self, model_gen, model_disc, optimizer_gen, optimizer_disc, scheduler_gen=None, scheduler_disc=None,
                 lambda_mel=1, lambda_wav=1):
        self.model_gen, self.model_disc = model_gen, model_disc
        self.optimizer_gen, self.optimizer_disc = optimizer_gen, optimizer_disc
        self.scheduler_gen, self.scheduler_disc = scheduler_gen, scheduler_disc
        # self.n_gen_steps = 1
        # self.epoch = 0


        self.gen_steps = 0
        self.disc_steps = 0

        #losses
        self.lambda_mel = lambda_mel
        self.lambda_wav = lambda_wav
        self.mel_loss = L1MelLoss()
        self.feature_loss = nn.L1Loss()

        self.gan_loss = nn.MSELoss()

    def train_step(self, train_loader):
        self.model_gen.train(True)
        self.model_disc.train(True)
        self.gen_loss = 0
        self.disc_loss = 0

        done_gen_steps = 0
        #         do_disc_step = True

        for batch_num, batch in enumerate(train_loader):
            mel = batch.melspec.to(device)
            wav = batch.waveform.to(device)

            # loss = step_disc(batch)

            # backward step
            disc_loss = self.step_disc(mel, wav)

            gen_loss = self.step_gen(mel, wav)

            #             if do_gen_step:

            self.gen_loss += gen_loss.item()
            self.disc_loss += disc_loss.item()


            if self.scheduler_disc is not None:
                self.scheduler_disc.step()
            if self.scheduler_disc is not None:
                self.scheduler_gen.step()

        return self.gen_loss / len(train_loader), self.disc_loss / len(train_loader)


    def step_gen(self, mel, wav):
        self.gen_steps += 1
        model_require_grad(self.model_disc, False)
        model_require_grad(self.model_gen, True)
        self.optimizer_gen.zero_grad()

        pred_wav = self.model_gen(mel).detach()
        gan_loss, feature_loss, rec_loss = self.calc_gen_loss(mel, wav, pred_wav)

        gen_loss = gan_loss + feature_loss + rec_loss
        gen_loss.backward()
        self.optimizer_gen.step()

        wandb.log({'train_gen_gan_loss': gan_loss.item(), 'train_gen_feature_loss':feature_loss.item(),
                   'train_gen_rec_loss': rec_loss.item(), 'train_gen_loss': gen_loss.item()})
        return gen_loss

    def step_disc(self, mel, wav):
        self.disc_steps += 1
        model_require_grad(self.model_disc, True)
        model_require_grad(self.model_gen, False)
        self.optimizer_disc.zero_grad()

        pred_wav = self.model_gen(mel).detach()

        pred_loss, real_loss = self.calc_disc_loss(mel, wav, pred_wav)

        disc_loss = pred_loss + real_loss
        disc_loss.backward()
        self.optimizer_disc.step()

        wandb.log({'train_disc_pred_loss': pred_loss.item(), 'train_disc_real_loss':real_loss.item(),
                   'train_disc_loss': disc_loss.item()})

        return disc_loss

    def calc_feature_loss(self, fm_pred, fm_real):
        loss = 0
        for preds, reals in zip(fm_pred, fm_real):
            for pred, real in zip(preds, reals):
                loss += self.feature_loss(pred, real)
        return loss


    def calc_gan_loss(self, msds, mpds, is_real):
        loss = 0
        for msd in msds:
            if is_real:
                target_msd = torch.ones_like(msd)
            else:
                target_msd = torch.zeros_like(msd)
            loss += self.gan_loss(msd, target_msd)
        for mpd in mpds:
            if is_real:
                target_mpd = torch.ones_like(mpd)
            else:
                target_mpd = torch.zeros_like(mpd)
            loss += self.gan_loss(mpd, target_mpd)


    def calc_gen_loss(self, mel, wav, pred_wav):
        if wav.shape[-1] < pred_wav.shape[-1]:
            wav = F.pad(wav, (0, pred_wav.shape[-1] - wav.shape[-1]))
        elif wav.shape[-1] > pred_wav.shape[-1]:
            pred_wav = F.pad(pred_wav, (0, wav.shape[-1] - pred_wav.shape[-1]))

        #mpd feature
        disc_mpd_pred, disc_mpd_pred_fm = self.model_disc.mpd(pred_wav.unsqueeze(1))
        disc_mpd_real, disc_mpd_real_fm = self.model_disc.mpd(wav.unsqueeze(1))

        mpd_feature_loss = self.calc_feature_loss(disc_mpd_pred_fm, disc_mpd_real_fm)

        # msd feature
        disc_msd_pred, disc_msd_pred_fm = self.model_disc.msd(pred_wav.unsqueeze(1))
        disc_msd_real, disc_msd_real_fm = self.model_disc.msd(wav.unsqueeze(1))

        msd_feature_loss = self.calc_feature_loss(disc_msd_pred_fm, disc_msd_real_fm)

        mel_loss = self.mel_loss(pred_wav, mel)

        gan_loss = self.calc_gan_loss(disc_msd_pred, disc_mpd_pred, is_real=True)

        return gan_loss, (msd_feature_loss +  mpd_feature_loss)* self.lambda_wav, mel_loss * self.lambda_mel

    def calc_disc_loss(self, mel, wav, pred_wav):
        if wav.shape[-1] < pred_wav.shape[-1]:
            wav = F.pad(wav, (0, pred_wav.shape[-1] - wav.shape[-1]))
        elif wav.shape[-1] > pred_wav.shape[-1]:
            pred_wav = F.pad(pred_wav, (0, wav.shape[-1] - pred_wav.shape[-1]))

        #mpd feature
        disc_mpd_pred, _ = self.model_disc.mpd(pred_wav.unsqueeze(1))
        disc_mpd_real, _ = self.model_disc.mpd(wav.unsqueeze(1))

        # msd feature
        disc_msd_pred, disc_msd_pred_fm = self.model_disc.msd(pred_wav.unsqueeze(1))
        disc_msd_real, disc_msd_real_fm = self.model_disc.msd(wav.unsqueeze(1))

        pred_loss = self.calc_gan_loss(disc_msd_pred, disc_mpd_pred, is_real=False)
        real_loss = self.calc_gan_loss(disc_msd_real, disc_mpd_real, is_real=True)

        return pred_loss, real_loss

