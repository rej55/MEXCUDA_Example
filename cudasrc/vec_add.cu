#include "mex.h"
#include "gpu/mxGPUArray.h"

__global__
void vec_add
(
    const double* src1,
    const double* src2,
    const double k1,
    const double k2,
    double* dst,
    int const N
)
{
    // Calculate index
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < N)
    {
        dst[tid] = k1*src1[tid] + k2*src2[tid];
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // Define variables
    const mxGPUArray *src1;
    const mxGPUArray *src2;
    double k1;
    double k2;
    mxGPUArray *dst;
    const double *d_src1;
    const double *d_src2;
    double *d_dst;
    int N1, N2;

    // Check the number of arguments
    if ( nrhs != 4 ) {
        mexErrMsgIdAndTxt("MATLAB:vec_add","The number of input arguments must be 4.");
    } 
    if ( nlhs != 1 ) {
        mexErrMsgIdAndTxt("MATLAB:vec_add","The number of output arguments must be 1.");
    } 

    // Initialization
    mxInitGPU();

    // Get data from *prhs[]
    src1 = mxGPUCreateFromMxArray(prhs[0]);
    src2 = mxGPUCreateFromMxArray(prhs[1]);
    k1 = mxGetScalar(prhs[2]);
    k2 = mxGetScalar(prhs[3]);

    // Check the dimension of src vectors
    N1 = (int)(mxGPUGetNumberOfElements(src1));
    N2 = (int)(mxGPUGetNumberOfElements(src2));
    if ( N1 != N2 ) {
        mxGPUDestroyGPUArray(src1);
        mxGPUDestroyGPUArray(src2);
        mexErrMsgIdAndTxt("MATLAB:vec_add","The dimension of input vectors must be same.");
    }

    // Get address of src1 and src2
    d_src1 = (const double*)(mxGPUGetDataReadOnly(src1));
    d_src2 = (const double*)(mxGPUGetDataReadOnly(src2));

    // Allocate memory of the destination variable on device memory
    dst = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(src1),
                            mxGPUGetDimensions(src1),
                            mxGPUGetClassID(src1),
                            mxGPUGetComplexity(src1),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_dst = (double *)(mxGPUGetData(dst));

    // Call kernel function
    dim3 block(N1);
    dim3 grid((N1 + block.x - 1) / block.x);
    vec_add<<<grid, block>>>(d_src1, d_src2, k1, k2, d_dst, N1);

    // Pass dst to plhs[0]
    plhs[0] = mxGPUCreateMxArrayOnGPU(dst);

    // Release memory
    mxGPUDestroyGPUArray(src1);
    mxGPUDestroyGPUArray(src2);
}