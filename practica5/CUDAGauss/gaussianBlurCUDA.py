from PIL import Image
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import scipy.stats as st
import time
import csv

MATRIX_SIZE = 1024
BLOCK_SIZE = 1024

class modeloCUDA:

    def __init__(self):
        print("***initialized***")    

    def gauss2D(self,shape=(3,3),sigma=3):
        
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        
        return h

    def gaussianBlurCUDA(self, sigma = 4, sizeMat = 3, imageFile = "media/CUDAGauss/índice.jpeg"):
        cuda.init()
        device = cuda.Device(0) # enter your gpu id here
        ctx = device.make_context()
        
        tamano=(MATRIX_SIZE,MATRIX_SIZE)
        foto=Image.open(imageFile).convert('L').resize(tamano)
        matriz=np.asarray(foto).astype(np.float32)

        matriz_gpu = cuda.mem_alloc(matriz.size * matriz.dtype.itemsize)
        cuda.memcpy_htod(matriz_gpu, matriz)

        original= foto

        matrizGauss=self.gauss2D(shape=(sizeMat,sizeMat),sigma=sigma)

        if sizeMat == 3:
            mod = SourceModule(f"""
            __global__ void filtro_gauss(float *matriz)
            {{

                int tx = blockDim.x*blockIdx.x + threadIdx.x;
                int ty = blockDim.y*blockIdx.y + threadIdx.y;
                float accu = 0;

                float a[3][3] = {{  
                    {{{matrizGauss[0,0]}, {matrizGauss[0,1]}, {matrizGauss[0,2]}}} ,
                    {{{matrizGauss[1,0]}, {matrizGauss[1,1]}, {matrizGauss[1,2]}}} ,   
                    {{{matrizGauss[2,0]}, {matrizGauss[2,1]}, {matrizGauss[2,2]}}}   
                }};

                float b[3][3] = {{
                    {{matriz[(tx-1025)+ty*1024], matriz[(tx-1024)+ty*1024], matriz[(tx-1023)+ty*1024]}},
                    {{matriz[(tx-1)+ty*1024], matriz[tx+ty*1024], matriz[(tx+1)+ty*1024]}},
                    {{matriz[(tx+1023)+ty*1024], matriz[(tx+1024)+ty*1024], matriz[(tx+1025)+ty*1024]}}
                }};

                if(blockIdx.x>0 && threadIdx.x>0 && blockIdx.x<1023 && threadIdx.x<1023) {{
                    
                    for (int i = 0; i < 3; i++) {{
                        for (int j = 0; j < 3; j++) {{
                            accu += a[i][j]*b[i][j];
                        }}
                    }}

                    matriz[tx+ty*1024]= accu;

                }}

            }}
            """)
        elif sizeMat == 5:
            mod = SourceModule(f"""
            __global__ void filtro_gauss(float *matriz)
            {{

                int tx = blockDim.x*blockIdx.x + threadIdx.x;
                int ty = blockDim.y*blockIdx.y + threadIdx.y;
                float accu = 0;

                float a[5][5] = {{  
                    {{{matrizGauss[0,0]}, {matrizGauss[0,1]}, {matrizGauss[0,2]}, {matrizGauss[0,3]}, {matrizGauss[0,4]}}} ,
                    {{{matrizGauss[1,0]}, {matrizGauss[1,1]}, {matrizGauss[1,2]}, {matrizGauss[1,3]}, {matrizGauss[1,4]}}} ,   
                    {{{matrizGauss[2,0]}, {matrizGauss[2,1]}, {matrizGauss[2,2]}, {matrizGauss[2,3]}, {matrizGauss[2,4]}}} , 
                    {{{matrizGauss[3,0]}, {matrizGauss[3,1]}, {matrizGauss[3,2]}, {matrizGauss[3,3]}, {matrizGauss[3,4]}}} ,
                    {{{matrizGauss[4,0]}, {matrizGauss[4,1]}, {matrizGauss[4,2]}, {matrizGauss[4,3]}, {matrizGauss[4,4]}}} 
                }};

                float b[5][5] = {{
                    {{matriz[(tx-2050)+ty*1024], matriz[(tx-2049)+ty*1024], matriz[(tx-2048)+ty*1024], matriz[(tx-2047)+ty*1024], matriz[(tx-2046)+ty*1024]}},
                    {{matriz[(tx-1026)+ty*1024], matriz[(tx-1025)+ty*1024], matriz[(tx-1024)+ty*1024], matriz[(tx-1023)+ty*1024], matriz[(tx-1022)+ty*1024]}},
                    {{matriz[(tx-2)+ty*1024], matriz[(tx-1)+ty*1024], matriz[tx+ty*1024], matriz[(tx+1)+ty*1024], matriz[(tx+2)+ty*1024]}},
                    {{matriz[(tx+1022)+ty*1024], matriz[(tx+1023)+ty*1024], matriz[(tx+1024)+ty*1024], matriz[(tx+1025)+ty*1024], matriz[(tx+1026)+ty*1024]}},
                    {{matriz[(tx+2046)+ty*1024], matriz[(tx+2047)+ty*1024], matriz[(tx+2048)+ty*1024], matriz[(tx+2049)+ty*1024], matriz[(tx+2050)+ty*1024]}}
                }};

                if(blockIdx.x>1 && threadIdx.x>1 && blockIdx.x<1022 && threadIdx.x<1022) {{
                    
                    for (int i = 0; i < 5; i++) {{
                        for (int j = 0; j < 5; j++) {{
                            accu += a[i][j]*b[i][j];
                        }}
                    }}

                    matriz[tx+ty*1024]= accu;

                }}

            }}
            """) 

        filtroGauss = mod.get_function("filtro_gauss")

        filtroGauss(
                matriz_gpu,
                block=(BLOCK_SIZE,1,1), grid = (BLOCK_SIZE,1), shared=0)

        aux = np.zeros_like(matriz).astype(np.float32)
        cuda.memcpy_dtoh(aux, matriz_gpu)
        ctx.pop()

        imagen_salida=Image.fromarray(np.uint8(aux))
        imagen_salida.save("media/CUDAGauss/result.jpg")
        

        