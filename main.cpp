#include<iostream>
#include<sys/stat.h>
#include<unistd.h>
#include<sys/mman.h>
#include<fcntl.h>


struct Config //Llama2.c导出的二进制文件头部的28个字节是配置参数
{
    int dim; //Transformer dimension
    int hidden_dim; //FFN层隐藏维度
    int n_layers; //层数
    int n_heads; //Query 头数
    int n_kv_heads; //Key/Value头数
    int vocab_size; //词表大小
    int seq_len; //最大序列长度
};

int main()
{
    const char* filename="stories15M.bin";

    struct stat sb; //用于描述文件的元数据
    if(stat(filename,&sb)==-1) //返回0表示正确填充，-1出错
    {
        std::cerr<<"Failed to stat file\n"; //无缓冲输出
        return 1;
    }

    int fd=open(filename,O_RDONLY); //O_RDONLY只读，O_WRONLY只写，O_RDWR可读可写
    if(fd==-1)
    {
        std::cerr<<"Failed to open file\n";
        return 1;
    }

    //mmap零CPU拷贝,访问数据时才由DMA把磁盘页加载到页缓存
    float* data=(float*)mmap(NULL,sb.st_size,PROT_READ,MAP_PRIVATE,fd,0); //各个参数分别是：期望映射到的虚拟地址、映射长度、内存访问权限（可读/可写/可执行）、映射类型标志（私有映射，修改不会同步回磁盘原文件）、文件描述符、文件内偏移量，成功返回映射区起始虚拟地址指针，失败返回-1
    if(data==MAP_FAILED)
    {
        std::cerr<<"mmap failed\n";
        return 1;
    }

    Config config;
    int* header=(int*)data;
    config.dim = header[0];
    config.hidden_dim = header[1];
    config.n_layers = header[2];
    config.n_heads = header[3];
    config.n_kv_heads = header[4];
    config.vocab_size = header[5];
    config.seq_len = header[6];

    std::cout<<"dim: "<<config.dim<<"\n"<<"vocab_size: "<<config.vocab_size<<"\n";

    float* weights_ptr=data+7;


    munmap(data,sb.st_size);
    close(fd);
    return 0;
}