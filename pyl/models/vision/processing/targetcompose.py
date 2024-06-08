from torchvision import transforms

class TargetCompose(transforms.Compose):
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, target):
        for t in self.transforms_list:
            target = t(target)
        return target
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms_list:
            format_string += '\n    {0},'.format(t)
        format_string += '\n)'
        return format_string

    def __str__(self):
        return self.__repr__()
