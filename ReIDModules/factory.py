from ReIDModules.AIM_model import AIMModel
from ReIDModules.CAL_model import CALModel


def ReID_module_factory(model_name, device):
    if model_name == 'CAL':
        return CALModel(device=device)
    if model_name == 'AIM':
        return AIMModel(device=device)
    if model_name == 'CTL':
        from ReIDModules.CTL_model import CTLModel
        return CTLModel(device=device)
