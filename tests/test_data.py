from vitquant.data.imagenette import IMAGENETTE_TO_IMAGENET1K


def test_class_mapping():
    # Imagenette wordnet-id sorted order -> ImageNet-1k indices
    assert IMAGENETTE_TO_IMAGENET1K == [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    assert len(set(IMAGENETTE_TO_IMAGENET1K)) == 10
    assert IMAGENETTE_TO_IMAGENET1K == sorted(IMAGENETTE_TO_IMAGENET1K)
