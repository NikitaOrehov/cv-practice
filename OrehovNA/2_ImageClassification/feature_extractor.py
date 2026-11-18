# feature_extractor.py
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

class BaseFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        pass

class SIFTExtractor(BaseFeatureExtractor):
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None:
            return np.array([])
        return descriptors  # shape: (N, 128)

class ORBExtractor(BaseFeatureExtractor):
    def __init__(self, nfeatures=500):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None:
            return np.array([])
        return descriptors  # shape: (N, 32)

class CNNExtractor(BaseFeatureExtractor):
    def __init__(self, model_name='resnet18', grid=3):
        super().__init__()
        self.grid = grid
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            self.model = torch.nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            self.model = torch.nn.Sequential(
                model.features,
                torch.nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            raise ValueError("Поддерживаются 'resnet18' или 'vgg16'")
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ph, pw = h // self.grid, w // self.grid
        patches = []
        for i in range(self.grid):
            for j in range(self.grid):
                patch = image[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                if patch.size == 0:
                    continue
                patches.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        if not patches:
            return np.array([])
        batch = torch.stack([self.transform(p) for p in patches])
        with torch.no_grad():
            feats = self.model(batch).flatten(1) 
        return feats.numpy()