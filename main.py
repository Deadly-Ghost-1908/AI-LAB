import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

thres = 0.50  
detection_running = False 
cap = cv2.VideoCapture("")
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(10, 70)
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
root = tk.Tk()
root.title("Object Detection App")
label = ttk.Label(root)
label.pack(padx=10, pady=10)
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output', 1280, 720)
def start_detection():
    global detection_running
    detection_running = True
    update()
def stop_detection():
    global detection_running
    detection_running = False
def get_latest_frame(cap):
    for _ in range(5):
        cap.read()
    return cap.read()
start_button = ttk.Button(root, text="Start Detection", command=start_detection)
start_button.pack(side=tk.LEFT, padx=10)
stop_button = ttk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack(side=tk.LEFT, padx=10)
def on_key_press(event):
    if event.char == 'q':
        root.destroy()
        stop_detection()
root.bind('<KeyPress>', on_key_press)
def update():
    global detection_running
    success, img = get_latest_frame(cap)
    if not success or img is None:
        return
    img = cv2.resize(img, (480, 360))
    if detection_running:
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if 0 <= classId < len(classNames):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                else:
                    print(f"Invalid classId: {classId}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
    label.img = img_tk
    label.config(image=img_tk)
    if detection_running:
        root.after(10, update)
root.mainloop()
cap.release()
cv2.destroyAllWindows()
