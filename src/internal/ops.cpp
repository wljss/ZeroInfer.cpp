#include "internal/ops.h"

#include <cmath>
#include <cstddef>

namespace zeroinfer::internal
{
void matmul(
    float* out,
    const float* x,
    const float* weights,
    int in_features,
    int out_features
)
{
    #pragma omp parallel for
    for(int i=0;i<out_features;++i)
    {
        const float* row=weights+static_cast<std::size_t>(i)*in_features;
        float value=0.0f;

        for(int j=0;j<in_features;++j)
        {
            value+=row[j]*x[j];
        }

        out[i]=value;
    }
}

void rmsnorm(float* out,const float* x,const float* weight,int size)
{
    float sum_squares=0.0f;

    for(int i=0;i<size;++i)
    {
        sum_squares+=x[i]*x[i];
    }

    const float scale=1.0f/std::sqrt(sum_squares/static_cast<float>(size)+1e-5f);

    for(int i=0;i<size;++i)
    {
        out[i]=weight[i]*(x[i]*scale);
    }
}

void softmax(float* values,int size)
{
    float max_value=values[0];

    for(int i=1;i<size;++i)
    {
        if(values[i]>max_value)
        {
            max_value=values[i];
        }
    }

    float sum=0.0f;
    for(int i=0;i<size;++i)
    {
        values[i]=std::exp(values[i]-max_value);
        sum+=values[i];
    }

    for(int i=0;i<size;++i)
    {
        values[i]/=sum;
    }
}
}
