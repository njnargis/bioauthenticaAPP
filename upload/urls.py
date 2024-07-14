from django.conf import settings
from django.conf.urls.static import static
from . import views
from .views import index,ImageUploadView
from django.urls import path
# urls.py
# backend/urls.py
urlpatterns = [
    path('backend/templates/upload/index.html',views.index, name='index'),
    path('', ImageUploadView.as_view(), name='image-upload'),
    path('video_stream/', views.video_stream, name='video_stream'),
    path('camera/', views.camera, name='camera'),
    #path('', views.webcam_view, name='webcam'),
    #path('video_feed/', views.video_feed, name='video_feed'),
    #path('capture_image/', views.capture_image, name='capture_image'),
    path('process_frame/', views.process_frame, name='process_frame'),
    path('video_capture/', views.video_capture, name='video_capture'),  # Add this line
   ] 