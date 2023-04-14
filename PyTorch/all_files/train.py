from functools import partial
import torch
from torchviz import make_dot_from_trace
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.tensorboard.graph import trace_model
from torchsummary import summary
import cv2
from joblib import cpu_count
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from ssd import build_ssd
from retinanet import build_retinanet
from load_config import get_config
from coco import get_dataset
from data import detection_collate
from trainer import Trainer
# from model_training.detection.ssd import build_ssd
# # from model_training.detection.retinanet_v2 import build_retinanet
# from model_training.detection.retinanet import build_retinanet
# from model_training.detection.config import get_config
# from model_training.detection.data import get_dataset, detection_collate
# from model_training.common.trainer import Trainer

cudnn.benchmark = False
cv2.setNumThreads(0)

import warnings
warnings.filterwarnings("ignore")



def _get_model(config):
    model_config = config['model']
    if model_config['name'] == 'ssd':
        model = build_ssd(model_config)
    elif model_config['name'] == 'retina_net':
        model = build_retinanet(model_config)
    else:
        raise ValueError("Model [%s] not recognized." % model_config['name'])
    return model


if __name__ == '__main__':
    #config = get_config("config/train.yaml")
    #config = get_config("C:\\Users\\KSHITIJ\\Desktop\\CLF\\University related\\Y4\\Final Year Project\\OriginalResearchCode\\AmurTigerCVWC\\PyTorch\\all_files\\new_train.yaml")
    config = get_config("C:\\Users\\KSHITIJ\\Desktop\\CLF\\University related\\Final Year Project\\OriginalResearchCode\\AmurTigerCVWC\\PyTorch\\all_files\\new_train.yaml")
    batch_size = config.pop('batch_size')
    get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=0,
                             shuffle=True, drop_last=True,
                             collate_fn=detection_collate, pin_memory=True)

    datasets = map(config.pop, ('train', 'val'))
    datasets = map(get_dataset, datasets)
    train, val = map(get_dataloader, datasets)
    # model = _get_model(config)
    # summary(model, input_size=(3, 448, 448))
    
    #trainer = Trainer(_get_model(config).cuda(), config, train=train, val=val)
    trainer = Trainer(_get_model(config), config, train=train, val=val)
    trainer.train()
