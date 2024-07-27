from django.db import models

# Create your models here.


class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    

class Comment(models.Model):
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
