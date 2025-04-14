from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from django.views.decorators.csrf import csrf_exempt

# Load your pretrained YOLOv8 model
model = YOLO('../assets/melanoma.pt')

@csrf_exempt
def predict(request):
    if request.method == 'POST' and request.FILES['image']:
        # print(model)
        # # Read the image file from the request
        image_file = request.FILES['image']
        image = np.fromstring(image_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # # Perform prediction
        results = model(image)
        # print(results)  
        # # Process results
        # # predictions = results.pandas().xyxy[0].to_dict(orient="records")
        predictions = []
        for result in results:
            for box in result.boxes:
                predictions.append({
                    # 'x,y': box.xyxy.item(),
                    # # 'y': box.xyxy.item(),
                    'confidence': box.conf.item(),
                    'class': box.cls.item()
                })
        if(len(predictions) > 1):
            
            class_1_predictions = [pred for pred in predictions if pred['class'] == 1]
            if class_1_predictions:
                predictions = [max(class_1_predictions, key=lambda x: x['confidence'])]
            else:
                predictions = [max(predictions, key=lambda x: x['confidence'])]
        print(predictions)


        return JsonResponse({'predictions': predictions})

    return JsonResponse({'error': 'Invalid request'}, status=400)