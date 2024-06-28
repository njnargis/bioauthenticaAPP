from django.conf import settings
from django.conf.urls.static import static
from . import views
from .views import index,ImageUploadView
from django.urls import path
# urls.py
# backend/urls.py
urlpatterns = [
    path('backend/templates/upload/index.html',index, name='index'),
    path('', ImageUploadView.as_view(), name='image-upload'),
   
   ] 