#if !defined CUDA_DISABLER

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y
#define bdx blockDim.x
#define bdy blockDim.y

#define BORDER_SIZE 5
#define MAX_KSIZE_HALF 100

namespace gblur
{
    __constant__ float c_gKer[MAX_KSIZE_HALF + 1];

    template <typename Border>
    __global__ void gaussianBlur(
            const int height, const int width, const PtrStepf src, const int ksizeHalf,
            const Border b, PtrStepf dst)
    {
        const int y = by * bdy + ty;
        const int x = bx * bdx + tx;

        extern __shared__ float smem[];
        volatile float *row = smem + ty * (bdx + 2*ksizeHalf);

        if (y < height)
        {
            // Vertical pass
            for (int i = tx; i < bdx + 2*ksizeHalf; i += bdx)
            {
                int xExt = int(bx * bdx) + i - ksizeHalf;
                xExt = b.idx_col(xExt);
                row[i] = src(y, xExt) * c_gKer[0];
                for (int j = 1; j <= ksizeHalf; ++j)
                    row[i] +=
                            (src(b.idx_row_low(y - j), xExt) +
                             src(b.idx_row_high(y + j), xExt)) * c_gKer[j];
            }

            if (x < width)
            {
                __syncthreads();

                // Horizontal pass
                row += tx + ksizeHalf;
                float res = row[0] * c_gKer[0];
                for (int i = 1; i <= ksizeHalf; ++i)
                    res += (row[-i] + row[i]) * c_gKer[i];
                dst(y, x) = res;
            }
        }
    }


    void setGaussianBlurKernel(const float *gKer, int ksizeHalf)
    {
        cudaSafeCall(cudaMemcpyToSymbol(c_gKer, gKer, (ksizeHalf + 1) * sizeof(*gKer)));
    }


    template <typename Border>
    void gaussianBlurCaller(const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, cudaStream_t stream)
    {
        int height = src.rows;
        int width = src.cols;

        dim3 block(256);
        dim3 grid(divUp(width, block.x), divUp(height, block.y));
        int smem = (block.x + 2*ksizeHalf) * block.y * sizeof(float);
        Border b(height, width);

        gaussianBlur<<<grid, block, smem, stream>>>(height, width, src, ksizeHalf, b, dst);

        cudaSafeCall(cudaGetLastError());

        if (stream == 0)
            cudaSafeCall(cudaDeviceSynchronize());
    }


    void gaussianBlurGpu(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, int borderMode, cudaStream_t stream)
    {
        typedef void (*caller_t)(const PtrStepSzf, int, PtrStepSzf, cudaStream_t);

        static const caller_t callers[] =
        {
            0 /*gaussianBlurCaller<BrdConstant<float> >*/,
            gaussianBlurCaller<BrdReplicate<float> >,
            0 /*gaussianBlurCaller<BrdReflect<float> >*/,
            0 /*gaussianBlurCaller<BrdWrap<float> >*/,
            gaussianBlurCaller<BrdReflect101<float> >
        };

        callers[borderMode](src, ksizeHalf, dst, stream);
    }


    template <typename Border>
    __global__ void gaussianBlur5(
            const int height, const int width, const PtrStepf src, const int ksizeHalf,
            const Border b, PtrStepf dst)
    {
        const int y = by * bdy + ty;
        const int x = bx * bdx + tx;

        extern __shared__ float smem[];

        const int smw = bdx + 2*ksizeHalf; // shared memory "width"
        volatile float *row = smem + 5 * ty * smw;

        if (y < height)
        {
            // Vertical pass
            for (int i = tx; i < bdx + 2*ksizeHalf; i += bdx)
            {
                int xExt = int(bx * bdx) + i - ksizeHalf;
                xExt = b.idx_col(xExt);

                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    row[k*smw + i] = src(k*height + y, xExt) * c_gKer[0];

                for (int j = 1; j <= ksizeHalf; ++j)
                    #pragma unroll
                    for (int k = 0; k < 5; ++k)
                        row[k*smw + i] +=
                                (src(k*height + b.idx_row_low(y - j), xExt) +
                                 src(k*height + b.idx_row_high(y + j), xExt)) * c_gKer[j];
            }

            if (x < width)
            {
                __syncthreads();

                // Horizontal pass

                row += tx + ksizeHalf;
                float res[5];

                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    res[k] = row[k*smw] * c_gKer[0];

                for (int i = 1; i <= ksizeHalf; ++i)
                    #pragma unroll
                    for (int k = 0; k < 5; ++k)
                        res[k] += (row[k*smw - i] + row[k*smw + i]) * c_gKer[i];

                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    dst(k*height + y, x) = res[k];
            }
        }
    }


    template <typename Border, int blockDimX>
    void gaussianBlur5Caller(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, cudaStream_t stream)
    {
        int height = src.rows / 5;
        int width = src.cols;

        dim3 block(blockDimX);
        dim3 grid(divUp(width, block.x), divUp(height, block.y));
        int smem = (block.x + 2*ksizeHalf) * 5 * block.y * sizeof(float);
        Border b(height, width);

        gaussianBlur5<<<grid, block, smem, stream>>>(height, width, src, ksizeHalf, b, dst);

        cudaSafeCall(cudaGetLastError());

        if (stream == 0)
            cudaSafeCall(cudaDeviceSynchronize());
    }


    void gaussianBlur5Gpu(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, int borderMode, cudaStream_t stream)
    {
        typedef void (*caller_t)(const PtrStepSzf, int, PtrStepSzf, cudaStream_t);

        static const caller_t callers[] =
        {
            0 /*gaussianBlur5Caller<BrdConstant<float>,256>*/,
            gaussianBlur5Caller<BrdReplicate<float>,256>,
            0 /*gaussianBlur5Caller<BrdReflect<float>,256>*/,
            0 /*gaussianBlur5Caller<BrdWrap<float>,256>*/,
            gaussianBlur5Caller<BrdReflect101<float>,256>
        };

        callers[borderMode](src, ksizeHalf, dst, stream);
    }

    void gaussianBlur5Gpu_CC11(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, int borderMode, cudaStream_t stream)
    {
        typedef void (*caller_t)(const PtrStepSzf, int, PtrStepSzf, cudaStream_t);

        static const caller_t callers[] =
        {
            0 /*gaussianBlur5Caller<BrdConstant<float>,128>*/,
            gaussianBlur5Caller<BrdReplicate<float>,128>,
            0 /*gaussianBlur5Caller<BrdReflect<float>,128>*/,
            0 /*gaussianBlur5Caller<BrdWrap<float>,128>*/,
            gaussianBlur5Caller<BrdReflect101<float>,128>
        };

        callers[borderMode](src, ksizeHalf, dst, stream);
    }

} // namespace gblur


#endif /* CUDA_DISABLER */
