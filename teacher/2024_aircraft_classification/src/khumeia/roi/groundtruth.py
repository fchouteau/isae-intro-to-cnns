from khumeia.roi.bounding_box import BoundingBox

__all__ = ["Groundtruth"]


class Groundtruth(BoundingBox):
    """
    A bbox with a label. Used to represent objects
    """

    def __init__(self, x_min: int, y_min: int, width: int, height: int, label: str):
        super(Groundtruth, self).__init__(x_min=x_min, y_min=y_min, width=width, height=height)
        self.label = label
