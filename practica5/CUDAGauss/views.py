from django.shortcuts import render
from .models import *
from .forms import *
from django.http import HttpResponseRedirect
from .gaussianBlurCUDA import modeloCUDA
import os
# Create your views here.
def subir_imagen(request):
    if request.method=='GET':
        return render(request,'subirImagen.html')
    elif request.method=="POST":
        form=ImageForm(request.POST,request.FILES)
        if form.is_valid():
            new_image=Image(image=form.cleaned_data["image"], name=form.cleaned_data["name"],opcion=form.cleaned_data["opcion"])            
            new_image.save()
            if form.cleaned_data["opcion"]:
                
                modeloCUDA().gaussianBlurCUDA(imageFile="media/CUDAGauss/"+str(form.cleaned_data["image"]),sizeMat = 5)
            else:
                
                modeloCUDA().gaussianBlurCUDA(imageFile="media/CUDAGauss/"+str(form.cleaned_data["image"]),sizeMat = 3)
            return HttpResponseRedirect('/CUDAGauss/filtroGauss/')
def ver_imagenes(request):
    imagen=Image.objects.last()      
    return render(request, "resultado.html",{'imagen':imagen})