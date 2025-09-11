from ultralytics import YOLO


model = YOLO("yolov12s.pt") #change the model using (n,s,m,l,x)

for result in model.predict(source=0, show=True, imgsz=640, device=0):
    print(result.boxes)  



#Bash : Run in cmd:
# After creating a virtual environment and installing ultralytics using pip install ultralytics


#yolo detect predict model=yolo12l.pt source=0 show=true