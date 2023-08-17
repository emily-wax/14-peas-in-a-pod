#include <cuda.h>
#include <stdio.h>

__global__ void hello( )
{
   printf("(CUDA) Hello World from the GPU !\n");
}

extern "C" void launch_kernel()
{

   hello<<< 1, 1 >>>();

   cudaDeviceSynchronize();

}
