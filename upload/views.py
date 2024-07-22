from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .serializers import ImageSerializer
from PIL import Image
import random
import subprocess
from imgaug import augmenters as iaa
from subprocess import run, PIPE

# Create your views here.
'''

def index(request):
    return render(request, 'upload/index.html')
'''
from django.http import JsonResponse
import numpy as np 
import cv2
from ultralytics import YOLO
import base64
import os
#from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
#from django.http import StreamingHttpResponse
import json
# Define paths to virtual environments
VENV_FINGERPRINT = "D:/G/python/fingerprint/ven"

def run_command(command, venv):
    """Run a command in a specified virtual environment."""
    activate_cmd = os.path.join(venv, 'Scripts', 'activate') + ' && '
    full_command = activate_cmd + command
    print(f"Running command: {full_command}")
    process = subprocess.run(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.stdout.decode('utf-8'), process.stderr.decode('utf-8')
# Load the model
model_path = 'D:/H/NEWREACTAPP/backend/backend/best.pt'  # Update this path
model = YOLO(model_path)
def camera(request):
    return render(request, 'upload/camera.html')
def js_to_image(js_object):
    image_bytes = base64.b64decode(js_object.split(',')[1])
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, flags=1)
    return img

def video_stream(request):
    if request.method == 'POST':
        data = request.POST['frame']
        img = js_to_image(data)

        # Perform inference
        results = model(img)

        # Process results
        response_data = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf.item()
                cls = int(box.cls.item())
                response_data.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': confidence, 'class': model.names[cls]
                })
        
        return JsonResponse(response_data, safe=False)
    return JsonResponse({'error': 'Invalid request method'}, status=400)
'''def webcam_view(request):
    return render(request, 'webcamapp/webcam.html')


def gen_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG codec

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def capture_image(request):
    if request.method == 'POST':
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return JsonResponse({'status': 'failed', 'message': 'Failed to capture image'}, status=400)

        _, buffer = cv2.imencode('.jpg', frame)
        image_data = ContentFile(buffer.tobytes(), name="captured_image.jpg")
        
        # Save or process the image_data here
        
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'failed'}, status=400)
    '''
def fingerprint_recognition_view(request):
    if request.method == 'POST':
        # Command to run the fingerprint recognition script
        command = "D:/G/python/fingerprint/ven/fingerprint_recognition.py"
        fingerprint_output, fingerprint_error = run_command(command, VENV_FINGERPRINT)
        # Parse output if necessary
       
        context = {
            'fingerprint_output': fingerprint_output,
            'fingerprint_error': fingerprint_error,
            
        }
        return JsonResponse(context)

    return JsonResponse({'message': 'Upload page'})
class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file = request.FILES.get('file')
        if file:
            # Save the file to a temporary location
            temp_file_path = 'temp_image.png'
            with open(temp_file_path, 'wb') as temp_file:
                for chunk in file.chunks():
                    temp_file.write(chunk)
            
            # Command to run the fingerprint recognition script
            command = f"python D:/G/pythonf/ingerprint/ven/fingerprint_recognition.py --image {temp_file_path}"
            fingerprint_output, fingerprint_error = run_command(command, VENV_FINGERPRINT)

            if fingerprint_error:
                return JsonResponse({'error': fingerprint_error}, status=500)
            
            result = json.loads(fingerprint_output)
            augmented_image = result['random_img']
            pred_rx = result['pred_rx']
            ry = result['ry']
            pred_ux = result['pred_ux']
            uy = result['uy']
            
            # Prepare the response
            augmented_image_base64 = base64.b64encode(augmented_image).decode('utf-8')
            
            response_data = {
                'augmented_image': augmented_image_base64,
                'pred_rx': pred_rx,
                'matched_label': ry,
                'pred_ux': pred_ux,
                'unmatched_label': uy
            }
            
            return JsonResponse(response_data)
        
        return JsonResponse({'error': 'No file provided'}, status=400)
               # views.py
def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str.split(',')[1])
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

@csrf_exempt
def process_frame(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        image_data = data.get('image')
        if image_data:
            img = base64_to_image(image_data)

            # Perform object detection
            results = model(img)

            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf.item()
                    cls = int(box.cls.item())
                    detections.append({
                        'class': model.names[cls],
                        'confidence': confidence,
                        'box': [x1, y1, x2, y2]
                    })

            return JsonResponse({'detections': detections})

    return JsonResponse({'error': 'Invalid request method'}, status=400)

def index(request):
    return render(request, 'upload/index.html')

def video_capture(request):
    return render(request, 'upload/index.html')