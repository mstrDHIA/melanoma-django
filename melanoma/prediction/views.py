from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
import torch
import cv2
import numpy as np
import json
import base64
from ultralytics import YOLO
from django.views.decorators.csrf import csrf_exempt

# Load your pretrained YOLOv8 model
model = YOLO('../assets/melanoma.pt')

@csrf_exempt
def predict(request):
    # Check if request method is POST and 'file' exists in request.FILES
    if request.method == 'POST':

        # First, check if the image was uploaded via `request.FILES`
        if 'file' in request.FILES:
            image_file = request.FILES['file']
            print("Image received via file upload")
        else:
            # If not, check if the image is being sent as base64 in the body
            try:
                body = json.loads(request.body.decode('utf-8'))
                image_data = body.get('image')
                if not image_data:
                    return JsonResponse({"error": "No image provided"}, status=400)
                print("Image received via base64")
                if image_data.startswith("data:image"):
                    # Extract the format and base64 string
                    format, imgstr = image_data.split(';base64,')
                    image_file = base64.b64decode(imgstr)
                else:
                    return JsonResponse({"error": "Invalid base64 image data"}, status=400)
            except Exception as e:
                return JsonResponse({"error": "Error processing base64 image: " + str(e)}, status=400)

        # At this point, `image_file` should contain the image data either from file or base64
        try:
            if isinstance(image_file, bytes):  # base64
                image = np.frombuffer(image_file, np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            else:  # file uploaded via request.FILES
                image = np.frombuffer(image_file.read(), np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            if image is None:
                return HttpResponseBadRequest("Invalid image file")

            # Perform prediction using YOLOv8 model
            results = model(image)
            predictions = []

            # Process results from YOLOv8
            for result in results:
                for box in result.boxes:
                    predictions.append({
                        'confidence': box.conf.item(),
                        'class': box.cls.item()
                    })

            # If there are multiple predictions, pick the one with the highest confidence
            if len(predictions) > 1:
                class_1_predictions = [pred for pred in predictions if pred['class'] == 1]
                if class_1_predictions:
                    predictions = [max(class_1_predictions, key=lambda x: x['confidence'])]
                else:
                    predictions = [max(predictions, key=lambda x: x['confidence'])]

            
            return JsonResponse({'predictions': predictions})

        except Exception as e:
            return HttpResponseBadRequest(f"Error processing the image: {str(e)}")

    return JsonResponse({'error': 'Invalid request method'}, status=405)
