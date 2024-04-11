from ultralytics import YOLO
    
# Load a model
model = YOLO('/home/wangzerui/code_all/ultralytics_yolov8/ultralytics/cfg/models/v8/yolo8z2.yaml')  # build a new model from YAML 
# model = YOLO('/home/wangzerui/code_all/ultralytics_yolov8/ultralytics/cfg/models/v8/yolov8.yaml')  
# model = YOLO('/home/wangzerui/code_all/ultralytics_yolov8/ultralytics/cfg/models/v8/yolo8z1.yaml')  # build a new model from YAML   
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
    
# Train the model
# results = model.train(data='ultralytics/cfg/datasets/dior.yaml', epochs=300, batch=16, device=5, name='dior_baseline1_batch16_SGD_epoch280')
# results = model.train(data='ultralytics/cfg/datasets/dior.yaml', epochs=300, batch=4, device=3, name='z1_')
# results = model.train(data='ultralytics/cfg/datasets/dior.yaml', epochs=300, batch=16, device=5, name='test')

results = model.train(data='/home/wangzerui/code_all/ultralytics_yolov8/ultralytics/cfg/datasets/dota1.0-h.yaml', epochs=300, batch=32, device=1, name='z1_')