from ultralytics import YOLO

class YOLOModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(YOLOModel, cls).__new__(cls)
            cls._instance.model = YOLO('./trained_models/best.pt')  

        return cls._instance
    
    def track(self, image):
        results = self.model.track(image,persist=True,conf=0.5)
        return results
    
    
