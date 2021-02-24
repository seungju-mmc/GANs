import torch
import datetime
import numpy as np
from WGAN.network import WGANnn
from WGAN.dataset import trainloader, testloader, imShow
from baseline.baseline.utils import jsonParser, getOptim
from torch.utils.tensorboard import SummaryWriter


class WGANTrainer:

    def __init__(
        self,
        path
    ):

        # parsing data from json file.
        parser = jsonParser(path)
        self.data = parser.loadParser()
        self.aData = self.data['GAN']
        self.optimData = self.data['optim']

        # Hyper-paramters
        self.clippingP = self.data['clippingP']
        self.batchSize = self.data['batchSize']
        self.nCritic = self.data['nCritic']
        self.epoch = self.data['epoch']
        device = self.data['device']
        self.device = torch.device(device)

        lPath = self.data['lPath']

        # GAN
        self.GAN = WGANnn(
            self.aData,
            device
        )

        if lPath:
            self.GAN.load_state_dict(
                torch.load(
                    lPath,
                    map_location=self.device
                )
            )

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
        with torch.no_grad():
            for prior in self.GAN.critic.priority:
                layerDict = self.GAN.critic.priorityModel[prior]
                for name in layerDict.keys():
                    parameters = layerDict[name].model.parameters()
                    for p in parameters:
                        _p = torch.clamp(
                            p,
                            -self.clippingP,
                            self.clippingP)
                        p.copy_(_p)

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

    def run(self):
        dataset = iter(trainloader)

        for epoch in range(self.epoch):
            for i in range(self.nCritic):
                for data in iter(dataset):
                    images, labels = data