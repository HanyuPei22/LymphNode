from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASE_PATH = PROJECT_ROOT

DATA_ROOT = BASE_PATH / 'data'
WATERMARK_PATH = BASE_PATH / 'data' / 'watermarks'
MODEL_ROOT = BASE_PATH / 'checkpoints' / 'clean_models'
RESULTS_ROOT = BASE_PATH / 'results'

for _dir in [DATA_ROOT, MODEL_ROOT, RESULTS_ROOT]:
    _dir.mkdir(parents=True, exist_ok=True)

PATHS = {
    'base': BASE_PATH,
    'data': DATA_ROOT,
    'watermarks': WATERMARK_PATH,
    'models': {
        'root': MODEL_ROOT,
        'clean': {
            'resnet18': MODEL_ROOT / 'clean_resnet18_cifar10_epoch.pth',
            'resnet50': MODEL_ROOT / 'clean_resnet50_cifar10_epoch100.pth',
            'vit_tiny': MODEL_ROOT / 'clean_vit_tiny_cifar10_epoch80.pth',
            'vit_small': MODEL_ROOT / 'clean_vit_small_cifar10_epoch80.pth',
        }
    },
    'results': RESULTS_ROOT,
}


def get_model_path(model_name, epoch=None):
    if epoch is None:
        return MODEL_ROOT / f'clean_{model_name}_epoch.pth'
    return MODEL_ROOT / f'clean_{model_name}_epoch{epoch}.pth'


def get_pattern_path(pattern_id=118):
    return WATERMARK_PATH / str(pattern_id) / 'pattern.npy'


def get_watermark_data_path(pattern_id, level):
    return WATERMARK_PATH / str(pattern_id) / f'solutions_{level}.npy'
