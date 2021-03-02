import torch
import datetime
import numpy as np
from WGAN.network import WGANnn
from WGAN.dataset import trainSet
from baseline.baseline.utils import jsonParser, getOptim
from torch.utils.tensorboard import SummaryWriter


class WGANTrainer:

    def __init__(
        self,
        path
    ):
        torch.cuda.empty_cache()

        # parsing data from json file.
        parser = jsonParser(path)
        self.data = parser.loadParser()
        self.aData = self.data['GAN']
        self.optimData = self.data['optim']

        # Hyper-paramters
        self.batchSize = self.data['batchSize']
        self.nCritic = self.data['nCritic']
        self.epoch = self.data['epoch']
        self._lambda = self.data['lambda']
        device = self.data['device']
        self.device = torch.device(device)
        lPath = self.data['lPath']

        # GAN
        self.GAN = WGANnn(
            self.aData,
            device,
            _lambda=self._lambda
        )
        # self.GAN.initParameters()

        if lPath:
            self.GAN.load_state_dict(
                torch.load(
                    lPath,
                    map_location=self.device
                )
            )
        self.GAN.to(self.device)
        self.genOptim()

        # save configuration
        name = "_WGAN_"
        sPath = self.data['sPath']
        time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.sPath = sPath + name + time + '.pth'

        # tensorboard configuration

        self.writeTMode = self.data['writeTMode']
        if self.writeTMode:
            tPath = self.data['tPath']
            self.tPath = tPath + name + time
            self.writer = SummaryWriter(self.tPath)
            self.writeTrainInfo()

    def genOptim(self):
        optimKeyList = list(self.optimData.keys())
        for optimKey in optimKeyList:
            if optimKey == "Critic":
                self.cOptim = getOptim(
                    self.optimData[optimKey],
                    self.GAN.critic.buildOptim()
                )

            if optimKey == "Generator":
                self.gOptim = getOptim(
                    self.optimData[optimKey],
                    self.GAN.generator.buildOptim()
                )

    def zeroGrad(self):
        self.cOptim.zero_grad()
        self.gOptim.zero_grad()

    def stepCritic(self, step, epoch):
        self.cOptim.step()
        normC = self.GAN.critic.calculateNorm().cpu().detach().numpy()
        if self.writeTMode:
            self.writer.add_scalar(
                'Critic Gradient Mag',
                normC, epoch * 100 + step
            )

    def stepGenerator(self, step, epoch):
        self.gOptim.step()
        normG = self.GAN.generator.calculateNorm().cpu().detach().numpy()
        if self.writeTMode:
            self.writer.add_scalar(
                'Generator Gradient Mag',
                normG,
                epoch * 100 + step
            )

    def writeDict(self, data, key, n=0):
        tab = ""
        for _ in range(n):
            tab += '\t'
        if type(data) == dict:
            for k in data.keys():
                dK = data[k]
                if type(dK) == dict:
                    self.info +=\
                        """
            {}{}:
                """.format(tab, k)
                    self.writeDict(dK, k, n=n+1)
                else:
                    self.info += \
                        """
            {}{}:{}
            """.format(tab, k, dK)
        else:
            self.info +=\
                    """
            {}:{}
            """.format(key, data)

    def writeTrainInfo(self):
        self.info = """
        Configuration for this experiment
        """
        key = self.data.keys()
        for k in key:
            data = self.data[k]
            if type(data) == dict:
                self.info +=\
                    """
            {}:
            """.format(k)
                self.writeDict(data, k, n=1)
            else:
                self.writeDict(data, k)

        print(self.info)
        if self.writeTMode:
            self.writer.add_text('info', self.info, 0)

    def spawnNoise(self, x, uniform=False):
        if uniform:
            zs = np.random.uniform(0, 1, x)
            zs = torch.tensor(zs).to(self.device).float()
            return zs
        # zs = np.random.normal(0, 1, 100 * x)
        zs = torch.randn(x, 100).to(self.device).float()
        # zs = torch.tensor(zs).to(self.device).float()
        # zs = zs.view(x, 100)
        return zs

    def run(self):
        dataset = torch.utils.data.DataLoader(
            trainSet,
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=8
        )
        step = 0
        Losses_g = 0
        Losses_c = 0
        Losses_grad = 0
        dataset.sampler
        zs_fake = self.spawnNoise(1)
        epoch = 0
        while 1:
            
            i = 0
            Loss = 0
            for data in iter(dataset):
                i += 1
                if i != self.nCritic:
                    # self.GAN.requireGradCritic()
                    step += 1
                    images, labels = data
                    images = images.to(self.device).detach()
                    bSize = len(images)
                    zs = self.spawnNoise(bSize).detach()
                    us = self.spawnNoise(bSize, uniform=True).detach()
                    Loss, gradientLoss = self.GAN.calCriticLoss(images, zs, us)
                    self.zeroGrad()
                    Loss.backward()
                    gradientLoss.backward()
                    self.stepCritic(step, epoch)
                    Losses_c += Loss.detach().cpu().numpy()
                    Losses_grad += gradientLoss.detach().cpu().numpy()
                    self.zeroGrad()
                else:
                    # self.GAN.requireGradCritic(False)
                    zs = self.spawnNoise(self.batchSize)
                    LossG = self.GAN.calGenLoss(zs)
                    self.zeroGrad()
                    LossG.backward()
                    self.stepGenerator(step, epoch)
                    self.zeroGrad()
                    Losses_g += LossG.detach().cpu().numpy()

                    print("""
                    epoch:{} // step:{} // Loss_G:{:.3f} // Loss_C:{:.3f}
                    """.format(epoch+1, step, Losses_g, Losses_c/self.nCritic))

                    if self.writeTMode:
                        self.writer.add_scalar(
                            "critic of Generator",
                            -Losses_g,
                            step)
                        self.writer.add_scalar(
                            "W Distance",
                            -Losses_c/self.nCritic,
                            step
                        )
                        self.writer.add_scalar(
                            "Gradient Penalty",
                            Losses_grad/self.nCritic,
                            step
                        )
                    Losses_g = 0
                    Losses_c = 0
                    Losses_grad = 0
                    i = 0

            with torch.no_grad():
                if self.writeTMode:
                    self.GAN.generator.evalMode()
                    fakeImage = self.GAN.genForward(zs_fake)[0]
                    fakeImage = fakeImage.cpu().mul(0.5).add(0.5)
                    self.writer.add_image(
                        "fakeImage",
                        fakeImage[0],
                        step)
                    
                    self.GAN.generator.trainMode()
            
            epoch += 1
