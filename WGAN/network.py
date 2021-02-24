import torch
from baseline.baseline.baseAgent import baseAgent


class WGANnn(torch.nn.Module):

    def __init__(
        self,
        aData,
        device,
        LSTMName=-1
    ):
        super(WGANnn, self).__init__()
        self.aData = aData
        self.keyList = list(self.aData.keys())
        self.device = torch.device(device)
        self.LSTMName = LSTMName
        self.buildModel()

    def buildModel(self):
        for netName in self.keyList:
            if netName == "Critic":
                netData = self.aData[netName]
                self.critic = baseAgent(
                    netData,
                    LSTMName=self.LSTMName
                )

            if netName == "Generator":
                netData = self.aData[netName]
                self.generator = baseAgent(
                    netData,
                    LSTMName=self.LSTMName
                )

    def criticForward(self, sample):
        critic = self.critic.forward(sample)[0]
        return critic

    def genForward(self, noise):
        gen = self.gen.forward(noise)[0]
        return gen

    def calCriticLoss(self, rState, noises):
        with torch.no_grad():
            gState = self.genForward(noises)

        rCritic = self.criticForward(rState)
        gCritic = self.criticForward(gState)

        Loss = torch.mean(gCritic - rCritic)
        

        return Loss

    def calGenLoss(self, noises):
        gState = self.genForward(noises)
        gCritic = self.criticForward(gState)
        Loss = -torch.mean(gCritic)

        return Loss

    def loadParameters(self):
        self.critic.loadParameters()
        self.generator.loadParameters()

    def to(self, device):
        self.critic.to(device)
        self.generator.to(device)