from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .serializers import ImageSerializer
# Create your views here.
'''

def index(request):
    return render(request, 'upload/index.html')
'''

class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
               # views.py


import os
import cv2
import uuid
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.http import JsonResponse

# Define the paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS_PATH = os.path.join(BASE_DIR, 'data', 'positive')
NEG_PATH = os.path.join(BASE_DIR, 'data', 'negative')
ANC_PATH = os.path.join(BASE_DIR, 'data', 'anchor')
INPUT_PATH = os.path.join(BASE_DIR, 'application_data', 'input_image', 'input_image.jpg')
VERIFICATION_PATH = os.path.join(BASE_DIR, 'application_data', 'verification_images')

# Ensure directories exist
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)
os.makedirs(os.path.dirname(INPUT_PATH), exist_ok=True)
os.makedirs(VERIFICATION_PATH, exist_ok=True)

def load_model():
    class L1Dist(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__()

        def call(self, input_embedding, validation_embedding):
            return tf.math.abs(input_embedding - validation_embedding)

    model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'siamesemodelv2.h5'), custom_objects={'L1Dist': L1Dist})
    return model

siamese_model = load_model()

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(VERIFICATION_PATH):
        input_img = preprocess(INPUT_PATH)
        validation_img = preprocess(os.path.join(VERIFICATION_PATH, image))
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(VERIFICATION_PATH))
    verified = verification > verification_threshold
    return results, verified

def index(request):
    return render(request, 'upload/index.html')

def verify_image(request):
    if request.method == 'POST':
        file = request.FILES['file']
        img_path = INPUT_PATH
        with open(img_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        results, verified = verify(siamese_model, 0.5, 0.5)
        response = {
            'verified': verified,
            'results': [float(res[0]) for res in results]
        }
        return JsonResponse(response)
    return JsonResponse({'error': 'Invalid request'}, status=400)
