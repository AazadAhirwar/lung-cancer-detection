from django.shortcuts import render
from django.http import HttpResponse
import cv2
from ultralytics import YOLO
import numpy as np
from django.core.files.base import ContentFile
import base64

model = YOLO("C:\\Users\\Admin\\OneDrive\\Desktop\\project_new\\model_project\\cancer\\segment_lungs_best.pt")


def predict_return(image):

    results = model.predict(image, save=False, save_txt=False)

    orig_img_array = results[0].orig_img.copy()

    class_index = results[0].boxes.cls.tolist()

    prec = results[0].boxes.conf.tolist()

    mask_cor = results[0].masks.xy

    print(class_index)
    class_names = results[0].names

    
    for box, class_id, p, m in zip(results[0].boxes.xyxy, class_index, prec, mask_cor):
       xmin, ymin, xmax, ymax = map(int, box)
       print(m)
       class_name = class_names[class_id]
       if p >= 0.48:
        cv2.rectangle(orig_img_array, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(orig_img_array, class_name+ str(p)[:4], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        mask_coords = np.array(m) 
        mask_coords = mask_coords.astype(np.int32)

        mask = np.zeros_like(orig_img_array)

        cv2.fillPoly(mask, [mask_coords], (255, 25, 255))  

        orig_img_array = cv2.addWeighted(orig_img_array, 1, mask, 0.5, 0)

    return orig_img_array



def home(request):

    
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        image_bytes = uploaded_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        output_image = predict_return(img)
        _, img_encoded = cv2.imencode('.png', output_image)
        img_data = base64.b64encode(img_encoded).decode()
        return render(request, 'home.html', {'uploaded_image': img_data})
    
    return render(request, 'home.html')



