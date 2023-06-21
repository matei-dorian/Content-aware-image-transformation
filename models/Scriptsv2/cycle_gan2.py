import torch
import torch.nn as nn
from torch.nn import init
import itertools
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
from torchvision.utils import save_image
from image_pool import ImagePool
from utils import *


class CycleGan:
    def __init__(self, opt):
        self.opt = opt
        self.current_epoch = opt.starting_epoch
        self.device = get_device()
        self.blur_layer = DecayingBlur()

        # networks
        self.G_A = Generator(num_channels=3, num_residuals=9).to(self.device)
        self.G_R = Generator(num_channels=3, num_residuals=9).to(self.device)
        self.D_A = Discriminator(in_channels=3).to(self.device)
        self.D_R = Discriminator(in_channels=3).to(self.device)

        # image pools for discriminators
        self.fake_A_pool = ImagePool(opt.pool_size)
        self.fake_R_pool = ImagePool(opt.pool_size)

        # optimizers
        self.optimizer_g = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_R.parameters()),
                                            lr=opt.learning_rate, betas=(opt.beta1, 0.999))
        self.optimizer_d = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_R.parameters()),
                                            lr=opt.learning_rate, betas=(opt.beta1, 0.999))

        # schedulers
        self.scheduler_g = get_scheduler(self.optimizer_g, self.current_epoch, self.opt.decay_epochs)
        self.scheduler_d = get_scheduler(self.optimizer_d, self.current_epoch, self.opt.decay_epochs)

        # grad scalers
        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

        # norms
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # images
        self.real = None
        self.anime = None
        self.fake_real = None
        self.fake_anime = None
        self.loss = {}

    def setup(self):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=0.02)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

        if not self.opt.resume_training:
            self.G_A.apply(init_func)
            self.G_R.apply(init_func)
            self.D_A.apply(init_func)
            self.D_R.apply(init_func)
            return

        # load the checkpoint if you want to continue training
        checkpoint_file = self.opt.models_root + f"checkpoint{self.opt.starting_epoch - 1}.pth"
        self.load_state(checkpoint_file)

    def forward(self, image_pair):
        self.real = image_pair[1].to(self.device)
        self.anime = image_pair[0].to(self.device)
        self.fake_anime = self.G_A(self.real)
        self.fake_real = self.G_R(self.anime)

    def backward_g(self):
        with torch.cuda.amp.autocast():
            # GAN loss
            disc_result_a = self.D_A(self.fake_anime)
            disc_result_r = self.D_R(self.fake_real)

            gan_loss_a = self.MSE(disc_result_a, torch.ones_like(disc_result_a))
            gan_loss_r = self.MSE(disc_result_r, torch.ones_like(disc_result_r))
            gan_loss = (gan_loss_a + gan_loss_r) * self.opt.lambda_gan
            self.loss["G_A_GAN"] = gan_loss_a
            self.loss["G_R_GAN"] = gan_loss_r

            # cycle loss
            cycle_a = self.G_A(self.fake_real)
            cycle_r = self.G_R(self.fake_anime)

            cyc_loss_a = self.L1(self.anime, cycle_a)
            cyc_loss_r = self.L1(self.real, cycle_r)
            cycle_loss = (cyc_loss_a + cyc_loss_r) * self.opt.lambda_cycle
            self.loss["G_A_Cycle"] = cyc_loss_a
            self.loss["G_R_Cycle"] = cyc_loss_r

            # identity loss
            id_loss = 0
            if self.opt.lambda_identity > 0:
              identity_a = self.G_A(self.anime)
              identity_r = self.G_R(self.real)

              id_loss_a = self.L1(self.anime, identity_a)
              id_loss_r = self.L1(self.real, identity_r)
              id_loss = (id_loss_a + id_loss_r) * self.opt.lambda_identity
              self.loss["G_A_id"] = id_loss_a
              self.loss["G_R_id"] = id_loss_r

            g_loss = gan_loss + cycle_loss + id_loss

        self.optimizer_g.zero_grad()
        self.g_scaler.scale(g_loss).backward()
        self.g_scaler.step(self.optimizer_g)
        self.g_scaler.update()

    def backward_d(self):
        fake_anime_pool = self.fake_A_pool.query(self.fake_anime)
        fake_real_pool = self.fake_R_pool.query(self.fake_real)

        with torch.cuda.amp.autocast():
            # D_A
            d_a_true = self.D_A(self.anime)
            d_a_false = self.D_A(fake_anime_pool.detach())

            d_a_true_loss = self.MSE(d_a_true, torch.ones_like(d_a_true))
            d_a_false_loss = self.MSE(d_a_false, torch.zeros_like(d_a_false))
            d_a_loss = d_a_true_loss + d_a_false_loss
            self.loss["D_A_loss"] = d_a_loss

            # D_R
            d_r_true = self.D_R(self.real)
            d_r_false = self.D_R(fake_real_pool.detach())

            d_r_true_loss = self.MSE(d_r_true, torch.ones_like(d_r_true))
            d_r_false_loss = self.MSE(d_r_false, torch.zeros_like(d_r_false))
            d_r_loss = d_r_true_loss + d_r_false_loss
            self.loss["D_R_loss"] = d_r_loss

            d_loss = (d_a_loss + d_r_loss) / 2

        self.optimizer_d.zero_grad()
        self.d_scaler.scale(d_loss).backward()
        self.d_scaler.step(self.optimizer_d)
        self.d_scaler.update()

    def train(self, loader):
        loop = tqdm(loader, leave=True)

        for idx, image_pair in enumerate(loop):
            anime_img = self.blur_layer(image_pair[0], self.current_epoch)
            real_img = self.blur_layer(image_pair[1], self.current_epoch)
            self.forward((anime_img, real_img))
            self.backward_d()
            self.backward_g()

            if idx % 700 == 0:
                save_image(self.real * 0.5 + 0.5, f"{self.opt.save_reals}/real{self.current_epoch}_{idx}.png")
                save_image(self.fake_anime * 0.5 + 0.5, f"{self.opt.save_reals}/fake{self.current_epoch}_{idx}.png")
                save_image(self.anime * 0.5 + 0.5, f"{self.opt.save_animes}/real{self.current_epoch}_{idx}.png")
                save_image(self.fake_real * 0.5 + 0.5, f"{self.opt.save_animes}/fake{self.current_epoch}_{idx}.png")

    def load_state(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.G_A.load_state_dict(checkpoint["Gen_A"])
        self.G_R.load_state_dict(checkpoint["Gen_R"])
        self.D_A.load_state_dict(checkpoint["Disc_A"])
        self.D_R.load_state_dict(checkpoint["Disc_R"])
        self.optimizer_g.load_state_dict(checkpoint["Optimizer_G"])
        self.optimizer_d.load_state_dict(checkpoint["Optimizer_D"])
        # self.scheduler_g.load_state_dict(checkpoint["Scheduler_G"])
        # self.scheduler_d.load_state_dict(checkpoint["Scheduler_D"])

        self.current_epoch = checkpoint["epoch"]

        lr = checkpoint["learning_rate"]
        for param_group in self.optimizer_g.param_groups:
            param_group["lr"] = lr
        for param_group in self.optimizer_d.param_groups:
            param_group["lr"] = lr

    def save_checkpoint(self, filename="checkpoint", extension=".pth"):
        checkpoint = {
            "epoch": self.current_epoch,
            "Gen_A": self.G_A.state_dict(),
            "Gen_R": self.G_R.state_dict(),
            "Disc_A": self.D_A.state_dict(),
            "Disc_R": self.D_R.state_dict(),
            "Optimizer_G": self.optimizer_g.state_dict(),
            "Optimizer_D": self.optimizer_d.state_dict(),
            "learning_rate": self.optimizer_g.param_groups[0]["lr"],
            # "Scheduler_G": self.scheduler_g.state_dit(),
            # "Scheduler_D": self.scheduler_d.state_dict()
        }
        torch.save(checkpoint, self.opt.models_root + filename + str(self.current_epoch) + extension)

    def update_learning_rate(self):
        self.scheduler_g.step()
        self.scheduler_d.step()
