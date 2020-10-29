# coding:utf-8
# !/usr/bin/python3


import os, sys
EXTs = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
################################################################################
def make_dataset(dir, cls_id, is_valid=None):
    if not callable(is_valid):
        if type(is_valid) in (str,tuple): EXTs = is_valid
        is_valid = lambda x: x.lower().endswith(EXTs)
    dir = os.path.expanduser(dir); data = []
    for k in sorted(cls_id.keys()):
        d = os.path.join(dir, k)
        if not os.path.isdir(d): continue
        for root, _, files in sorted(os.walk(d)):
            for ff in sorted(files):
                ff = os.path.join(root, ff)
                if is_valid(ff): data.append((ff, cls_id[k]))
    return data


def find_classes(dir): # ensure: no class in subdir of another
    if sys.version_info >= (3,5): # Faster for Python 3.5+
        cls = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        cls = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]
    cls.sort(); cls_id = {k:id for id,k in enumerate(cls)}; return cls, cls_id


from torchvision.datasets import VisionDataset # for torchvision>=0.3.0
################################################################################
class FromFolder(VisionDataset):
    """ Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(self, root, loader, transform=None, target_transform=None, is_valid=None):
        super(FromFolder, self).__init__(root, transform=transform,
                                         target_transform=target_transform)
        self.classes, self.class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, self.class_to_idx, is_valid)
        assert len(samples)>0, "No valid data found."; self.loader = loader
        self.samples = samples; self.targets = [s[1] for s in samples]
        self.extensions = is_valid if type(is_valid) in (str,tuple) else EXTs

    def _find_classes(self, dir): return find_classes(dir)

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target # class_index


################################################################################
