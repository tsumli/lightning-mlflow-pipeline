from abc import abstractmethod

from torchvision import transforms


class TrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, (1., 1.), (1., 1.)),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(.5, .5, .5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __call__(self, img):
        img = img.convert('RGB') 
        a = self.transform(img)
        return a
    
class TestTransform(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, (1., 1.), (1., 1.)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __call__(self, img):
        img = img.convert('RGB') 
        a = self.transform(img)
        return a

