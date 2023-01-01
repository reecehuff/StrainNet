#%% Imports
import torch.nn as nn

#-- Scripts 
from core.utils import load_model
import core.utils as utils

class StrainNet(nn.Module):
    def __init__(self, args):
        super(StrainNet, self).__init__()
        self.args = args

        # Load the DefomationClassifier
        self.DeformationClassifier = load_model(args, 'DeformationClassifier')

        # Load TensionNet
        self.TensionNet = load_model(args, 'TensionNet')

        # Load CompressionNet
        self.CompressionNet = load_model(args, 'CompressionNet')

        # Load RigidNet
        self.RigidNet = load_model(args, 'RigidNet')

    
    def forward(self, imgs):
        
        # Begin by predicting the deformation type
        deformation_type = self.DeformationClassifier(imgs)
        deformation_type = utils.class2deformation(deformation_type)

        # If the deformation type is tension, then use the TensionNet
        if deformation_type == 'tension':
            strain = self.TensionNet(imgs)
        # If the deformation type is compression, then use the CompressionNet
        elif deformation_type == 'compression':
            strain = self.CompressionNet(imgs)
        # If the deformation type is rigid, then use the RigidNet
        elif deformation_type == 'rigid':
            strain = self.RigidNet(imgs)

        return strain