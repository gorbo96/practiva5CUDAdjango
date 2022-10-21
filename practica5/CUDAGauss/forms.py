from django import forms
from django.forms.fields import ImageField
from CUDAGauss import models
class ImageForm(forms.ModelForm):
    class Meta:
        model=models.Image
        fields=['image','name','opcion']        