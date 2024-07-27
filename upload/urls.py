from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views
from .views import index,ImageUploadView, save_comment,get_comments,delete_comment

# urls.py
# backend/urls.py
urlpatterns = [
    path('backend/templates/upload/index.html',views.index, name='index'),
    path('', ImageUploadView.as_view(), name='image-upload'),
    path('video_stream/', views.video_stream, name='video_stream'),
    path('camera/', views.camera, name='camera'),
    path('process_frame/', views.process_frame, name='process_frame'),
    path('video_capture/', views.video_capture, name='video_capture'),  # Add this line
    path('upload/', views.fingerprint_recognition_view, name='upload'),
    path('save_image/', views.save_image, name='save_image'),
    path('save_comment/', save_comment, name='save_comment'),
    path('get_comments/', get_comments, name='get_comments'),
    path('delete_comment/<int:comment_id>/', delete_comment, name='delete_comment'),
   ] 