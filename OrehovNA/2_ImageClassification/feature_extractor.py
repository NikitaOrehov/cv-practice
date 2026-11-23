# feature_extractor.py
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

class BaseFeatureExtractor(ABC):

    _registry: dict[str, type] = {}

    def __init_subclass__(cls, register_name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if register_name:
            BaseFeatureExtractor._registry[register_name] = cls

    @classmethod
    def create(cls, name: str, grid: int = 4, layer: str | None = None,
              transform_size: int | None = None, **extra) -> "BaseFeatureExtractor":
        normalized = name.lower()
        if normalized not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unknown extractor '{name}'. Available: {available}")

        if normalized in ('sift', 'orb', 'kaze'):
            return cls._registry[normalized](**extra)

        kwargs = {'grid': grid}
        if layer is not None:
            kwargs['layer_name'] = layer
        if transform_size is not None:
            kwargs['transform_cfg'] = {
                'size': (transform_size, transform_size),
                'mean': [0.485, 0.456, 0.406],
                'std':  [0.229, 0.224, 0.225],
            }
        kwargs.update(extra)
        return cls._registry[normalized](**kwargs)

    @classmethod
    def available_methods(cls) -> list[str]:
        return sorted(cls._registry)

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        ...

    @property
    def descriptor_dim(self) -> int:
        return -1


class ClassicalExtractor(BaseFeatureExtractor, ABC):

    def __init__(self):
        self._detector = self._create_detector()

    @abstractmethod
    def _create_detector(self):
        pass

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        if descriptors is None:
            return np.array([])
        return descriptors


class SIFTExtractor(ClassicalExtractor, register_name="sift"):
    def _create_detector(self):
        return cv2.SIFT_create()

    @property
    def descriptor_dim(self) -> int:
        return 128


class ORBExtractor(ClassicalExtractor, register_name="orb"):
    def __init__(self, nfeatures: int = 500):
        self.nfeatures = nfeatures
        super().__init__()

    def _create_detector(self):
        return cv2.ORB_create(nfeatures=self.nfeatures)

    @property
    def descriptor_dim(self) -> int:
        return 32


class KAZEExtractor(ClassicalExtractor, register_name="kaze"):

    def _create_detector(self):
        return cv2.KAZE_create()

    @property
    def descriptor_dim(self) -> int:
        return 64




DEFAULT_TRANSFORM_CFG = {
    'resnet18': {'size': (224, 224),
                 'mean': [0.485, 0.456, 0.406],
                 'std':  [0.229, 0.224, 0.225]},
    'resnet50': {'size': (224, 224),
                 'mean': [0.485, 0.456, 0.406],
                 'std':  [0.229, 0.224, 0.225]},
    'vgg16':    {'size': (224, 224),
                 'mean': [0.485, 0.456, 0.406],
                 'std':  [0.229, 0.224, 0.225]},
    'convnext_base': {'size': (224, 224),
                      'mean': [0.485, 0.456, 0.406],
                      'std':  [0.229, 0.224, 0.225]},
    'dinov2':   {'size': (224, 224),
                 'mean': [0.485, 0.456, 0.406],
                 'std':  [0.229, 0.224, 0.225]},
}

DEFAULT_LAYER = {
    'resnet18': 'layer3',
    'resnet50': 'layer3',
    'vgg16':    'features',
    'convnext_base': 'avgpool',
    'dinov2':   'norm',
}



class NeuralExtractor(BaseFeatureExtractor, ABC):
    def __init__(self, model_name: str, grid: int = 4,
                 layer_name: str = None, transform_cfg: dict = None):
        self.model_name = model_name
        self.grid = grid
        self.layer_name = layer_name or DEFAULT_LAYER.get(model_name, 'default')
        cfg = transform_cfg or DEFAULT_TRANSFORM_CFG.get(model_name)
        if cfg is None:
            raise ValueError(f"Нет transform_cfg для '{model_name}'")
        self.transform_cfg = cfg

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._captured = None

        self.model = self._build_model().to(self.device).eval()
        self.transform = self._build_transform()
        self._register_hook()

    def extract(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ph, pw = h // self.grid, w // self.grid
        if ph == 0 or pw == 0:
            return np.array([])

        patches = []
        for i in range(self.grid):
            for j in range(self.grid):
                patch = image[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                if patch.size == 0:
                    continue
                patches.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        if not patches:
            return np.array([])

        batch = torch.stack([self.transform(p) for p in patches]).to(self.device)
        with torch.no_grad():
            _ = self.model(batch)
        feats = self._captured
        if feats is None:
            raise RuntimeError("Forward hook не сработал — проверьте layer_name")
        if feats.dim() == 4:
            feats = feats.mean(dim=(2, 3))
        return feats.cpu().numpy().astype(np.float32)

    def _build_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.transform_cfg['size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.transform_cfg['mean'],
                                 std=self.transform_cfg['std']),
        ])

    def _register_hook(self):
        target = self._resolve_layer()
        if target is None:
            raise ValueError(f"Слой '{self.layer_name}' не найден в {self.model_name}")
        target.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._captured = output

    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def _resolve_layer(self) -> torch.nn.Module:
        pass


class ResNet18Extractor(NeuralExtractor, register_name="resnet18"):
    def __init__(self, model_name='resnet18', **kwargs):
        super().__init__(model_name, **kwargs)

    def _build_model(self):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return torch.nn.Sequential(*list(model.children())[:-1])

    def _resolve_layer(self):
        idx = {'layer1': 4, 'layer2': 5, 'layer3': 6, 'layer4': 7}.get(self.layer_name)
        if idx is not None:
            return self.model[idx]
        if self.layer_name == 'avgpool':
            return self.model[-1]
        raise ValueError(f"Неподдерживаемый слой ResNet18: {self.layer_name}")

    @property
    def descriptor_dim(self) -> int:
        return {'layer1': 64, 'layer2': 128, 'layer3': 256,
                'layer4': 512, 'avgpool': 512}.get(self.layer_name, -1)


class ResNet50Extractor(NeuralExtractor, register_name="resnet50"):
    def __init__(self, model_name='resnet50', **kwargs):
        super().__init__(model_name, **kwargs)

    def _build_model(self):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        return torch.nn.Sequential(*list(model.children())[:-1])

    def _resolve_layer(self):
        idx = {'layer1': 4, 'layer2': 5, 'layer3': 6, 'layer4': 7}.get(self.layer_name)
        if idx is not None:
            return self.model[idx]
        if self.layer_name == 'avgpool':
            return self.model[-1]
        raise ValueError(f"Неподдерживаемый слой ResNet50: {self.layer_name}")

    @property
    def descriptor_dim(self) -> int:
        return {'layer1': 256, 'layer2': 512, 'layer3': 1024,
                'layer4': 2048, 'avgpool': 2048}.get(self.layer_name, -1)


class VGG16Extractor(NeuralExtractor, register_name="vgg16"):
    def __init__(self, model_name='vgg16', **kwargs):
        super().__init__(model_name, **kwargs)

    def _build_model(self):
        base = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        return torch.nn.Sequential(
            base.features,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

    def _resolve_layer(self):
        if self.layer_name == 'features':
            return self.model[0]
        idx = {'conv1_2': 2, 'conv2_2': 5, 'conv3_3': 10,
               'conv4_3': 17, 'conv5_3': 28}.get(self.layer_name)
        if idx is None:
            raise ValueError(f"Неизвестный слой VGG16: {self.layer_name}")
        return self.model[0][idx]

    @property
    def descriptor_dim(self) -> int:
        return {'conv3_3': 256, 'conv4_3': 512, 'conv5_3': 512,
                'features': 512}.get(self.layer_name, -1)

class ConvNeXtBaseExtractor(NeuralExtractor, register_name="convnext_base"):
    def __init__(self, model_name='convnext_base', **kwargs):
        super().__init__(model_name, **kwargs)

    def _build_model(self):
        base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        # Берем features + avgpool, без финального классификатора
        return torch.nn.Sequential(
            base.features,
            base.avgpool,
        )

    def _resolve_layer(self):
        if self.layer_name == 'avgpool':
            return self.model[1]
        if self.layer_name == 'stage3':
            return self.model[0][5]
        if self.layer_name == 'stage4':
            return self.model[0][7]
        raise ValueError(f"Неподдерживаемый слой ConvNeXt: {self.layer_name}")

    @property
    def descriptor_dim(self) -> int:
        return {'stage3': 512, 'stage4': 1024, 'avgpool': 1024}.get(self.layer_name, -1)


class DINOv2Extractor(NeuralExtractor, register_name="dinov2"):
    """
    Визуальный трансформер DINOv2 (ViT-B/14) от Meta AI.
    Идеально подходит для поиска и классификации достопримечательностей.
    """
    def __init__(self, model_name='dinov2', **kwargs):
        super().__init__(model_name, **kwargs)

    def _build_model(self):
        return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    def _resolve_layer(self):
        if self.layer_name == 'norm':
            return self.model.norm
        raise ValueError(f"Неподдерживаемый слой DINOv2: {self.layer_name}")

    def extract(self, image: np.ndarray) -> np.ndarray:
        # Переопределяем extract, т.к. ViT возвращает 3D-тензор (B, N_tokens, 768),
        # где 0-й токен — это глобальный CLS-токен патча.
        h, w = image.shape[:2]
        ph, pw = h // self.grid, w // self.grid
        if ph == 0 or pw == 0:
            return np.array([])

        patches = []
        for i in range(self.grid):
            for j in range(self.grid):
                patch = image[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                if patch.size == 0:
                    continue
                patches.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        if not patches:
            return np.array([])

        batch = torch.stack([self.transform(p) for p in patches]).to(self.device)
        with torch.no_grad():
            _ = self.model(batch)
        feats = self._captured
        if feats is None:
            raise RuntimeError("Forward hook не сработал для DINOv2")
        # Берем CLS-токен (индекс 0 по оси токенов) -> (num_patches, 768)
        cls_feats = feats[:, 0, :]
        return cls_feats.cpu().numpy().astype(np.float32)

    @property
    def descriptor_dim(self) -> int:
        return 768
