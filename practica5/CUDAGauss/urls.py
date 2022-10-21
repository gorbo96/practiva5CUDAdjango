from django.urls import path
from django.urls.resolvers import URLPattern
from CUDAGauss import views
app_name='CUDAGauss'
urlpatterns=[path('subirImagen/',views.subir_imagen,name='subir_imagen'),
path('filtroGauss/',views.ver_imagenes,name='ver_imagenes')
]