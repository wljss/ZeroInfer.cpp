#pragma once

namespace zeroinfer::internal
{
void matmul(
    float* out,
    const float* x,
    const float* weights,
    int in_features,
    int out_features
);

void rmsnorm(float* out,const float* x,const float* weight,int size);

void softmax(float* values,int size);
}
