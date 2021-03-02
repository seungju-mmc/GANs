import torch
from baseline.baseline.baseAgent import baseAgent


class WGANnn(torch.nn.Module):

    def __init__(
        self,
        aData,
        device,
        _lambda=10,
        LSTMName=-1
    ):
        super(WGANnn, self).__init__()
        self.aData = aData
        self.keyList = list(self.aData.keys())
        self.device = torch.device(device)
        self.LSTMName = LSTMName
        self._lambda = _lambda
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
        if type(sample) is not tuple:
            sample = tuple([sample])
        critic = self.critic.forward(sample)
        return critic

    def genForward(self, noise):
        if type(noise) is not tuple:
            noise = tuple([noise])
        gen = self.generator.forward(noise)
        return gen

    def calCriticLoss(self, rState, noises, uNoises):

        # real
        rState = torch.autograd.Variable(rState)
        rCritic = self.criticForward(rState)[0].mean()

        # fake
        noises = torch.autograd.Variable(noises, volatile=True)
        gState = self.genForward(noises)[0]
        gState = torch.autograd.Variable(gState.data)
        gCritic = self.criticForward(gState)[0].mean()
        # gCritic = torch.autograd.Variable(gCritic)

        Loss = gCritic - rCritic

        uNoises = torch.unsqueeze(uNoises, dim=-1)
        uNoises = torch.unsqueeze(uNoises, dim=-1)
        uNoises = torch.unsqueeze(uNoises, dim=-1).detach()
        mState = rState.data * uNoises + (1 - uNoises) * gState[0].data
        mState = torch.autograd.Variable(mState, requires_grad=True)
        mCritic = self.criticForward(mState)[0]
        grad = torch.autograd.grad(
            mCritic,
            mState,
            grad_outputs=torch.ones(
                mCritic.size()
            ).to(self.device),
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )[0]
        gradient = grad.view(grad.size(0), -1)
        gradient = gradient.norm(2, dim=1)
        gradient_Loss = self._lambda * ((gradient - 1)**2).mean()

        return Loss, gradient_Loss

    def calGenLoss(self, noises):
        noises = torch.autograd.Variable(noises)
        gState = self.genForward(noises)
        gCritic = self.criticForward(gState)[0]
        Loss = -torch.mean(gCritic)

        return Loss

    def loadParameters(self):
        self.critic.loadParameters()
        self.generator.loadParameters()

    def initParameters(self):
        with torch.no_grad():
            for prior in self.critic.priority:
                layerDict = self.critic.priorityModel[prior]
                for name in layerDict.keys():
                    parameters = layerDict[name].model.parameters()
                    for p in parameters:
                        torch.nn.init.normal_(p, 0, std=0.2)

            for prior in self.generator.priority:
                layerDict = self.generator.priorityModel[prior]
                for name in layerDict.keys():
                    parameters = layerDict[name].model.parameters()
                    for p in parameters:
                        torch.nn.init.normal_(p, 0, std=0.2)

    def to(self, device):
        self.critic.to(device)
        self.generator.to(device)
    
    def requireGradCritic(self, cond=True):
        for prior in self.critic.priority:
            layerDict = self.critic.priorityModel[prior]
            for name in layerDict.keys():
                parameters = layerDict[name].model.parameters()
                for p in parameters:
                    p.requires_grad = cond
