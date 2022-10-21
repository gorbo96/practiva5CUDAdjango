from django.db import models

class Image(models.Model):
    image=models.ImageField(upload_to='CUDAGauss',default='CUDAGauss/static/images/no-img.jpg')
    name=models.CharField(max_length=200)
    opcion=models.BooleanField()
