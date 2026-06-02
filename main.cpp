#include<iostream>
#include<vector>
#include<sys/stat.h>
#include<unistd.h>
#include<sys/mman.h>
#include<fcntl.h>
#include<array>
#include<fstream>
#include<cstdint>
#include<string>

struct Config //Llama2.c导出的二进制文件头部的28个字节是配置参数
{
    int dim; //Transformer dimension
    int hidden_dim; //FFN层隐藏维度
    int n_layers; //层数
    int n_q_heads; //Query 头数
    int n_kv_heads; //Key/Value 头数,GQA和上面头数不一样
    int vocab_size; //词表大小
    int seq_len; //最大序列长度
};
struct TransformerWeights //网络权重
{
    float* token_embedding_table; //维度为(vocab_size,dim)，词嵌入表
    float* rms_att_weight; //维度为(n_layers, dim)，attention子层前面的RMSNorm权重
    float* rms_ffn_weight; //维度为(n_layers, dim)，FFN子层前面的RMSNorm权重
    float* Wq; //维度为(n_layers, dim, dim( = n_q_heads * head_size))
    float* Wk; //维度为(n_layers, dim, n_kv_heads * head_size)
    float* Wv; //维度为(n_layers, dim, n_kv_heads * head_size)
    float* Wo; //维度为(n_layers, dim, dim( = n_q_heads * head_size))
    float* W1; //维度为(n_layers, hidden_dim , dim),SwiGLU上投影矩阵,把dim维投影到hidden_dim维
    float* W2; //维度为(n_layers, dim , hidden_dim),SwiGLU下投影矩阵,把FFN中间维度投影回Transformer主干维度
    float* W3; //维度为(n_layers, hidden_dim , dim),SwiGLU上投影矩阵,gate分支
    float* rms_final_weight; //所有Transformer层结束后的最终RMSNorm权重
    float* wcls; //最终分类器权重，把最终hidden state转成词表大小的logits
};

struct RunState
{
    // vector相比new[]可以自己释放，性能上也差不多
    std::vector<float> x; //TODO:
    std::vector<float> xb; //
    std::vector<float> xb2; //
    std::vector<float> hb; //
    std::vector<float> hb2; //
    std::vector<float> q; //
    float* k = nullptr; //后面会指向cache
    float* v = nullptr; //后面会指向cache
    std::vector<float> att; //
    std::vector<float> logits; //
    std::vector<float> key_cache; //
    std::vector<float> value_cache; //

    RunState()=default; //默认构造函数
    explicit RunState(const Config &p){allocate(p);} //带参数的构造函数。explicit的作用是禁止隐式类型转换
    
    void allocate(const Config &p)
    {
        const int kv_dim = p.dim / p.n_q_heads * p.n_kv_heads ;

        x.assign(static_cast<size_t>(p.dim), 0.0f); //TODO:这些维度怎么来的？
        xb.assign(static_cast<size_t>(p.dim), 0.0f);
        xb2.assign(static_cast<size_t>(p.dim), 0.0f);

        hb.assign(static_cast<size_t>(p.hidden_dim), 0.0f);
        hb2.assign(static_cast<size_t>(p.hidden_dim), 0.0f);

        q.assign(static_cast<size_t>(p.dim), 0.0f);

        key_cache.assign(static_cast<size_t>(p.n_layers) * p.seq_len * kv_dim, 0.0f);
        value_cache.assign(static_cast<size_t>(p.n_layers) * p.seq_len * kv_dim, 0.0f);

        att.assign(static_cast<size_t>(p.n_q_heads) * p.seq_len, 0.0f);

        logits.assign(p.vocab_size, 0.0f);
    }
};

struct Transformer
{
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd; //checkpoint文件的文件描述符
    float* data; //指向checkpoint文件映射到内存的起点
    ssize_t file_size; //checkpoint文件字节大小

    ~Transformer() 
    {
        if (data != nullptr && data != MAP_FAILED) 
        {
            munmap(data, file_size);
        }
        if (fd != -1) 
        {
            close(fd);
        }
    }
};

void assign_weights(TransformerWeights &weights,float* weights_ptr,const Config &config,const bool shared_weights)
{
    const size_t n_layers = static_cast<size_t>(config.n_layers); //这样做是为了防止int会溢出
    //赋值权重
    weights.token_embedding_table = weights_ptr;
    weights_ptr += config.vocab_size * config.dim;

    weights.rms_att_weight = weights_ptr;
    weights_ptr += n_layers * config.dim;

    int head_size=config.dim / config.n_q_heads; //head_size为每个头的维度
    weights.Wq = weights_ptr;
    weights_ptr += n_layers * config.dim * config.dim;

    weights.Wk = weights_ptr;
    weights_ptr += n_layers * config.dim * (config.n_kv_heads * head_size);

    weights.Wv = weights_ptr;
    weights_ptr += n_layers * config.dim * (config.n_kv_heads * head_size);

    weights.Wo = weights_ptr;
    weights_ptr += n_layers * config.dim * config.dim;

    weights.rms_ffn_weight = weights_ptr;
    weights_ptr += n_layers * config.dim;

    weights.W1 = weights_ptr;
    weights_ptr += n_layers * config.hidden_dim * config.dim;

    weights.W2 = weights_ptr;
    weights_ptr += n_layers * config.dim * config.hidden_dim;

    weights.W3 = weights_ptr;
    weights_ptr += n_layers * config.hidden_dim * config.dim;

    weights.rms_final_weight = weights_ptr;
    weights_ptr += config.dim;
    const size_t rope_table_size = static_cast<size_t>(config.seq_len) * head_size / 2;

    // 跳过的是旧的ROPE用到的部分，这里用不到了
    weights_ptr += rope_table_size;
    weights_ptr += rope_table_size;

    weights.wcls = weights.token_embedding_table;// TODO:源码好像用的是share部分

}

struct TokenIndex //用来根据字符串找到token id
{
    std::string str;
    int id;
};

struct Tokenizer 
{
    std::vector<std::string> vocab; //词表数组，根据token id找到字符串
    std::vector<float> vocab_scores; //用于BPE合并
    std::vector<TokenIndex> sorted_vocab; //按照字符串排序，encode阶段能快速根据字符串找到token id

    int vocab_size = 0; //词表大小
    std::uint32_t max_token_length = 0; //词表中最长token字符串长度TODO:

    std::array<unsigned char, 512> byte_pieces{}; //储存256个单字节字符串，每个字符串占2个字节，例如'A'后面接一个'\0'
};

bool compare_tokens(const TokenIndex& a, const TokenIndex& b) { // 按照字典序排序
    return a.str < b.str;
}

void build_tokenizer(Tokenizer& tokenizer, const std::string& tokenizer_path, int vocab_size) 
{
    //初始化
    tokenizer.vocab.resize(vocab_size);
    tokenizer.vocab_scores.resize(vocab_size);
    tokenizer.sorted_vocab.clear();
    tokenizer.vocab_size = vocab_size;

    for (int i = 0; i < 256; i++) {
        tokenizer.byte_pieces[i * 2] = static_cast<unsigned char>(i);
        tokenizer.byte_pieces[i * 2 + 1] = '\0';
    }

    //读文件
    std::ifstream file(tokenizer_path, std::ios::binary);
    if (!file) 
    {
        throw std::runtime_error("couldn't load " + tokenizer_path);
    }

    auto read_exact = [&](char* dst, std::size_t size) //lambda表达式,这个&把外部变量按引用捕获进来了
    {
        if (!file.read(dst, static_cast<std::streamsize>(size))) //static_cast是显示普通类型转换
        {
            throw std::runtime_error("failed read");
        }
    };

    std::int32_t max_len_from_file = 0;
    read_exact(reinterpret_cast<char*>(&max_len_from_file), sizeof(std::int32_t)); //reinterpret_cast是底层内存重新解释的转换
    tokenizer.max_token_length = static_cast<std::uint32_t>(max_len_from_file);

    for (int i = 0; i < vocab_size; i++) 
    {
        read_exact(reinterpret_cast<char*>(&tokenizer.vocab_scores[i]), sizeof(float));

        std::int32_t len = 0;
        read_exact(reinterpret_cast<char*>(&len), sizeof(std::int32_t));

        if (len < 0) 
        {
            throw std::runtime_error("invalid token length");
        }

        std::string token(len, '\0');
        read_exact(&token[0], static_cast<std::size_t>(len));

        tokenizer.vocab[i] = std::move(token); //通过move移动赋值，不写的话就是拷贝赋值
    }
}

struct ProbIndex // top-p采样时用
{
    float prob; //token的概率
    int index; //token的id
}; 

struct Sampler //采样器，把模型输出的logits转换成下一个token
{
    int vocab_size; //词表大小
    std::vector<ProbIndex> probindex; //top-p采样时用的临时数组
    float temperature; //改变logits分布的尖锐程度，logits除以temperature，temperature越小越尖锐保守，temperature越大越平坦随机
    float topp; //核采样参数
    unsigned long long rng_state; //随机数生成器的状态

    Sampler()=default; //默认构造函数
    explicit Sampler(int vocab_size_,float temperature_,float topp_,unsigned long long rng_seed_) : vocab_size(vocab_size_),probindex(vocab_size_),temperature(temperature_),topp(topp_),rng_state(rng_seed_)//带参数的构造函数。explicit的作用是禁止隐式类型转换
    {
    } 
};

void matmul(float* xout,const float* x,const float* w,int n,int d) // W(d,n)矩阵乘法x(n,)得到的结果存到xout (d,)
{
    #pragma omp parallel for //并行计算i
    for(int i=0;i<d;++i) 
    {
        const float* row=w+i*n; //行号

        float val=0.0f;
        for (int j=0;j<n;++j) //j是串行计算的 
        {
            val+=row[j]*x[j];
        }

        xout[i]=val;
    }
}
void generate(Transformer &transformer,Tokenizer &tokenizer,Sampler &sampler,const std::string &prompt,int steps)
{
    printf("Welcome!");
}

bool read_checkpoint_header_and_map(Config &config,const std::string filename,float* &data,int &fd,ssize_t &file_size)
{
    struct stat sb; //用于描述文件的元数据
    if(stat(filename.c_str(),&sb)==-1) //返回0表示正确填充，-1出错
    {
        throw std::runtime_error("Failed to stat file\n"); 
    }

    fd=open(filename.c_str(),O_RDONLY); //O_RDONLY只读，O_WRONLY只写，O_RDWR可读可写
    if(fd==-1)
    {
        throw std::runtime_error("Failed to open file\n"); 
    }

    //mmap零CPU拷贝,访问数据时才由DMA把磁盘页加载到页缓存
    file_size=sb.st_size;
    data=(float*)mmap(NULL,file_size,PROT_READ,MAP_PRIVATE,fd,0); //各个参数分别是：期望映射到的虚拟地址、映射长度、内存访问权限（可读/可写/可执行）、映射类型标志（私有映射，修改不会同步回磁盘原文件）、文件描述符、文件内偏移量，成功返回映射区起始虚拟地址指针，失败返回-1
    if(data==MAP_FAILED)
    {
        throw std::runtime_error("mmap failed\n"); 
    }


    int* header=(int*)data;
    config.dim = header[0];
    config.hidden_dim = header[1];
    config.n_layers = header[2];
    config.n_q_heads = header[3];
    config.n_kv_heads = header[4];
    config.vocab_size = header[5];
    config.seq_len = header[6];

    std::cout<<"dim: "<<config.dim<<"\n"<<"vocab_size: "<<config.vocab_size<<"\n";

    return 1;
}
void build_transformer(Transformer& transformer, const std::string& checkpoint_path)
{
    bool shared_weights = read_checkpoint_header_and_map(transformer.config,checkpoint_path,transformer.data,transformer.fd,transformer.file_size);

    transformer.state.allocate(transformer.config);

    float* weights_ptr = transformer.data + sizeof(Config) / sizeof(float);

    assign_weights(transformer.weights,weights_ptr,transformer.config,shared_weights); //赋值模型权重
}
int main()
{
    Transformer transformer;
    build_transformer(transformer,"stories15M.bin");

    Tokenizer tokenizer;
    build_tokenizer(tokenizer,"tokenizer.bin",transformer.config.vocab_size);
    
    Sampler sampler(transformer.config.vocab_size,0.95,0.9,233333); //构造采样器

    generate(transformer,tokenizer,sampler,"Long long ago",transformer.config.seq_len);

    return 0;
}