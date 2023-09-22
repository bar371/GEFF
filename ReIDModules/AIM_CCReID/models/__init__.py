import logging
from ReIDModules.AIM_CCReID.models.classifier import Classifier, NormalizedClassifier
from ReIDModules.AIM_CCReID.models.img_resnet import ResNet50
from ReIDModules.AIM_CCReID.models.PM import PM
from ReIDModules.AIM_CCReID.models.Fusion import Fusion

def build_model(config, num_identities, num_clothes):
    logger = logging.getLogger('reid.model')

    # Build backbone
    logger.info("Initializing model: Resnet50, model2: HPM")

    model = ResNet50(config)
    model2 = PM(feature_dim=config.MODEL.FEATURE_DIM)
    fusion = Fusion(feature_dim=config.MODEL.FEATURE_DIM)
    
    logger.info("Model  size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    logger.info("Model2 size: {:.5f}M".format(sum(p.numel() for p in model2.parameters())/1000000.0))

    # Build classifier
    identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
    clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)
    clothes_classifier2 = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)

    return model, model2, fusion, identity_classifier, clothes_classifier, clothes_classifier2