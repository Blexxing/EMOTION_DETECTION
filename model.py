# model.py
from fer import FER

def get_detector():
    """
    Returns a FER detector using MTCNN for face detection
    """
    return FER(mtcnn=True)
