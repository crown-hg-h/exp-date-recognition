import albumentations
from albumentations.pytorch import ToTensorV2


def _get_base_augmentations():
    return [
        albumentations.LongestMaxSize(667),
        albumentations.SmallestMaxSize(400),
        albumentations.Rotate(5, p=0.5),
        albumentations.RandomBrightnessContrast(p=0.5),
        albumentations.GaussianBlur(p=0.3)
    ]


def _get_test_augmentations():
    return [
        albumentations.LongestMaxSize(667),
        albumentations.SmallestMaxSize(400),
        albumentations.Normalize(),
        ToTensorV2()
    ]


def _get_train_augmentation():
    return _get_base_augmentations() + [albumentations.Normalize(), ToTensorV2(p=1.0)]


def get_viz_transform():
    return albumentations.Compose(
        _get_base_augmentations(),
        bbox_params=albumentations.BboxParams(
            format='pascal_voc', label_fields=['categories'])
    )


def get_train_transform():
    return albumentations.Compose(
        _get_train_augmentation(),
        bbox_params=albumentations.BboxParams(
            format='pascal_voc', label_fields=['categories'])
    )


def get_test_transform():
    return albumentations.Compose(
        _get_test_augmentations(),
        bbox_params=albumentations.BboxParams(
            format='pascal_voc', label_fields=['categories'])
    )


def get_detect_transform():
    return albumentations.Compose(_get_test_augmentations())
