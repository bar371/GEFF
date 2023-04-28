from ReIDModules.CAL_model import CALModel


def ReID_module_factory(model_name, device):
    if model_name == 'CAL':
        return CALModel(device=device)
