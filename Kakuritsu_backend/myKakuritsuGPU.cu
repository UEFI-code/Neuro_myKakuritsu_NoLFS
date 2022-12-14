// MIT License

// Copyright (c) Microsoft Corporation and SuperHacker UEFI.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

__global__ void myCell_forward_kernel(const float* input, const float* weight, float Kakuritsu, float* output, const int Neuros, const int InputDim, const unsigned int timeNow) 
{
	//Here InputDim == NumberOfSynapses
	const int CellID = threadIdx.x;
	const int BatchID = blockIdx.x;
	const float *myWeightBase = weight + CellID * InputDim;
	//const float *myKakuriBase = Kakuritsu + CellID * InputDim;
	const float *myInputBase = input + BatchID * InputDim;
	float *myOutput = output + BatchID * Neuros + CellID;
	
	*myOutput = 0.0;

	curandState RandState;
	curand_init(timeNow, CellID + BatchID, 0, &RandState);

	for(int i=0; i<InputDim; i++)
	{
		if(curand_uniform(&RandState) < Kakuritsu)
		    *myOutput += myWeightBase[i] * myInputBase[i];
	}

	return;
}

__global__ void myKasoCell_backward_kernel(const float* input, const float* weight, float* output, const int KasoNeuros, const int InputDim)
{
	//Here InputDim == RealCellNumber, KasoNeuros == NumberOfSynapses
	const int KasoCellID = threadIdx.x;
	//KasoCellID match RealCell's pin
        const int BatchID = blockIdx.x;

	const float *myInput = input + BatchID * InputDim;
	const float *myWeight = weight + KasoCellID;
	float *myOutput = output + KasoCellID + BatchID * KasoNeuros;
	*myOutput = 0.0;

	for(int i = 0; i < InputDim; i++)
	{
		*myOutput += myWeight[i * KasoNeuros] * myInput[i];
	}

	return;
}

template <typename scalar_t>
__global__ void ms_demo_matmul_kernel(
    const scalar_t* A,
    const scalar_t* B,
    scalar_t* C,
    const int M, 
    const int K, 
    const int N,
    const bool trans_A = false,
    const bool trans_B = false) 
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N)
    {
        scalar_t sum = 0.0;
        for (int k = 0; k < K; k++)
        {
            const int i = trans_A ? (k * M + row) : (row * K + k);
            const int j = trans_B ? (col * K + k) : (k * N + col);
            sum += A[i] * B[j];
        }

        C[row * N + col]  = sum;
    }
}

std::vector<torch::Tensor> myKakuritsu_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    float Kakuritsu)
{
    const int Batchsize = input.size(0);
    const int InputDim = input.size(1);
    const int Neuros = weights.size(0);

    auto output = torch::zeros({Batchsize, Neuros}, torch::TensorOptions().device(torch::kCUDA));

    float *pGPUinput = input.data<float>();
    float *pGPUweights = weights.data<float>();
    //float *pGPUKakuritsu = Kakuritsu.data<float>(); 
    float *pGPUoutput = output.data<float>();

    myCell_forward_kernel<<<Batchsize, Neuros>>>(pGPUinput, pGPUweights, Kakuritsu, pGPUoutput, Neuros, InputDim, (unsigned int)time(NULL));

    return {output};
}

std::vector<torch::Tensor> myKakuritsu_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights)
{
    const int Batchsize = grad_output.size(0);
    const int RealCellNum = grad_output.size(1);
    const int KasoCellNum = weights.size(1);

    auto grad_input = torch::zeros({Batchsize, KasoCellNum}, torch::TensorOptions().device(torch::kCUDA));
    auto grad_weights = torch::zeros({RealCellNum, KasoCellNum}, torch::TensorOptions().device(torch::kCUDA));

    const dim3 block(32, 32);
    const dim3 grid1((Batchsize - 1) / 32 + 1, (KasoCellNum - 1) / 32 + 1);
    const dim3 grid2((RealCellNum - 1) / 32 + 1, (KasoCellNum - 1) / 32 + 1);

    float *pGPUgrad_input = grad_input.data<float>();
    float *pGPUgrad_weights = grad_weights.data<float>();
    float *pGPUgrad_output = grad_output.data<float>();
    float *pGPUinput = input.data<float>();
    float *pGPUweights = weights.data<float>();

    myKasoCell_backward_kernel<<<Batchsize, KasoCellNum>>>(pGPUgrad_output, pGPUweights, pGPUgrad_input, KasoCellNum, RealCellNum);
    ms_demo_matmul_kernel<float><<<grid2, block>>>(pGPUgrad_output, pGPUinput, pGPUgrad_weights, RealCellNum, Batchsize, KasoCellNum, true, false);
    
    return {grad_input, grad_weights};
}
