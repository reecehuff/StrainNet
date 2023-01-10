#%% Imports
import torch.nn as nn

#-- Scripts 
from core import utils

class StrainNet(nn.Module):
    def __init__(self, args):
        super(StrainNet, self).__init__()
        self.args = args

        # Load the DefomationClassifier
        self.DeformationClassifier = utils.load_model(args, 'DeformationClassifier')

        # Load TensionNet
        self.TensionNet = utils.load_model(args, 'TensionNet')

        # Load CompressionNet
        self.CompressionNet = utils.load_model(args, 'CompressionNet')

        # Load RigidNet
        self.RigidNet = utils.load_model(args, 'RigidNet')
    
    def forward(self, imgs):
        
        # Begin by predicting the deformation type
        deformation_type = self.DeformationClassifier(imgs)
        deformation_class = utils.softmax_to_class(deformation_type)
        deformation_name = utils.get_deformation_type(deformation_class)

        # If the deformation type is tension, then use the TensionNet
        if deformation_name == 'Tension':
            strain = self.TensionNet(imgs)
        # If the deformation type is compression, then use the CompressionNet
        elif deformation_name == 'Compression':
            strain = self.CompressionNet(imgs)
        # If the deformation type is rigid, then use the RigidNet
        elif deformation_name == 'Rigid':
            strain = self.RigidNet(imgs)

        return strain, deformation_class.item()