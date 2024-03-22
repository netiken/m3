/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
#include "topo.h"

const double MTU=1000.0;
const double DELAY_PROP_BASE=1000.0;
const double BYTE_TO_BIT=8.0;
const double HEADER_SIZE=48.0;
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct
{
    int input_dim;        // transformer dimension
    int hidden_dim_1; // for ffn layers
    int hidden_dim_2;   // number of layers
    int output_dim;    // max sequence length
    int y_len;    // max sequence length
} ConfigMLP;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim) -> embedding to linear layer
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} ModelWeightsTransformer;

typedef struct
{
    float *w1; // (hidden_dim, dim)
    float *w2; // (hidden_dim, hidden_dim)
    float *w3; // (dim, hidden_dim)
    float *const_opt; // const placeholder for optimization
} ModelWeightsMLP;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct
{
    float *x;      // activation at current time stamp (dim,)
    float *h1;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float *h2;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float *logits; // output logits
} RunStateMLP;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    ModelWeightsTransformer weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

typedef struct {
    uint32_t flowId, src, dst;
    double fat;
    double size;
    uint32_t idx;
    double remaining_size;
    double fct_est;
    float bw_bottleneck;
    float sldn;
} Flow;


void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void malloc_run_state_mlp(RunStateMLP *s, ConfigMLP *p)
{
    // we calloc instead of malloc to keep valgrind happy
    s->x = calloc(p->input_dim, sizeof(float));
    s->h1 = calloc(p->hidden_dim_1, sizeof(float));
    s->h2 = calloc(p->hidden_dim_2, sizeof(float));
    s->logits = calloc(p->output_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->h1 || !s->h2 || !s->logits)
    {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void free_run_state_mlp(RunStateMLP *s)
{
    free(s->x);
    free(s->h1);
    free(s->h2);
    free(s->logits);
}

void memory_map_weights(ModelWeightsTransformer *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim + p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void memory_map_weights_mlp(ModelWeightsMLP *w, ConfigMLP *p, float *f)
{
    float *ptr = f;
    w->w1 = ptr;
    ptr += p->input_dim * p->hidden_dim_1 + p->hidden_dim_1;

    w->w2 = ptr;
    ptr += p->hidden_dim_1 * p->hidden_dim_2 + p->hidden_dim_2;

    w->w3 = ptr;
    ptr += p->output_dim * p->hidden_dim_2 + p->output_dim;

    w->const_opt = ptr;
    ptr += p->y_len;
}

void read_checkpoint(char* checkpoint, Config* config, ModelWeightsTransformer* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* ckpt_path_llama) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(ckpt_path_llama, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void build_mlp(char* ckpt_path_mlp, ConfigMLP *config_mlp, ModelWeightsMLP *weights_mlp, RunStateMLP *state_mlp) {
    // read in the Config and the Weights from the checkpoint
    int fd = 0;
    float *data = NULL;
    long file_size;

    FILE *file = fopen(ckpt_path_mlp, "rb");
    if (!file){ fprintf(stderr, "Couldn't open file %s\n", ckpt_path_mlp); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config_mlp, sizeof(ConfigMLP), 1, file) != 1) { exit(EXIT_FAILURE); }
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    file_size = ftell(file);  // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    fd = open(ckpt_path_mlp, O_RDONLY); // open in read only mode
    if (fd == -1){ fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float *weights_ptr = data + sizeof(ConfigMLP) / sizeof(float);
    memory_map_weights_mlp(weights_mlp,config_mlp, weights_ptr);
    // create and init the application RunState
    malloc_run_state_mlp(state_mlp, config_mlp);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul_with_bias(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val + w[d * n + i];
        // xout[i] = val;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, float* feat,int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    ModelWeightsTransformer* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    // printf("feat-%d: %f, %f\n ",pos, feat[0],feat[p->vocab_size-2]);
    // printf("weight-%d: %f, %f\n ",pos, w->token_embedding_table[0],w->token_embedding_table[p->vocab_size*p->dim-1]);
    // printf("bias-%d: %f, %f\n ",pos, w->token_embedding_table[p->vocab_size*p->dim],w->token_embedding_table[p->vocab_size*p->dim+p->dim-1]);
    matmul_with_bias(x, feat, w->token_embedding_table, p->vocab_size, p->dim);
    // printf("token_embedding_table-%d: %f, %f\n ",pos, x[0],x[p->dim-1]);
    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->dim);
    return s->logits;
}

void forward_MLP(float *x, ConfigMLP *p, ModelWeightsMLP *w, RunStateMLP *s)
{
// #pragma omp parallel for
    matmul_with_bias(s->h1, x, w->w1, p->input_dim, p->hidden_dim_1);
    for (int i = 0; i < p->hidden_dim_1; i++)
    {
        if (s->h1[i]<0.0){
            s->h1[i]=0.0;
        }
    }

    matmul_with_bias(s->h2, s->h1, w->w2, p->hidden_dim_1, p->hidden_dim_2);
    for (int i = 0; i < p->hidden_dim_2; i++)
    {
        if (s->h2[i]<0.0){
            s->h2[i]=0.0;
        }
    }

    matmul_with_bias(s->logits, s->h2, w->w3, p->hidden_dim_2, p->output_dim);
    
    //ReLU activation function for the output layer
    // for (int i = 0; i < p->output_dim; i++)
    // {
    //     if (s->logits[i]<0.0){
    //         s->logits[i]=0.0;
    //     }
    // }

    // No activation function for the output layer
    // for (int i = 0; i < p->output_dim; i++)
    // {
    //     s->logits[i] =(1.0f / (1.0f + exp(-s->logits[i])));
    // }
}


unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop
void generate(Transformer *transformer, float* feat_map, int n_hosts, int n_feat_input,int n_feat_context, float* feat_concat) {
    // start the main loop
    // long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int pos = 0;     // position in the sequence
    float *feat_input = (float*)malloc((n_feat_input) * sizeof(float));
    float* feat_start = feat_map + n_feat_input;
    while (pos < n_hosts-1) {
        memcpy(feat_input, feat_start, n_feat_input*sizeof(*feat_input));
        // printf("logits-%d: %f, %f\n ",pos, feat_input[0],feat_input[n_feat_input-2]);
        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, feat_input,pos);
        // printf("logits-%d: %f, %f\n ",pos, logits[0],logits[n_feat_context-1]);
        feat_start+=n_feat_input;
        pos++;
        for (int i = 0; i < n_feat_context; i++) {
            feat_concat[n_feat_input+i] += logits[i];
        }

    }
    for (int i = 0; i < n_feat_context; ++i) {
        feat_concat[n_feat_input+i] /= pos;
    }
    // printf("\n");
    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    // long end = time_in_ms();
    // fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    free(feat_input);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

void get_fct_mmf(unsigned int n_flows, Flow* flows, int h, int *topo, int method_mmf, int method_routing, int type_topo, int base_lr);
void update_rate_mmf(unsigned int traffic_count, int *src, int *dst, int method_mmf, int type_topo);

void update_rate_mmf(unsigned int traffic_count, int *src, int *dst, int method_mmf, int type_topo)
{
    int iteration_count = 0;
    double exec_time = 0.0;
    
    pl_ppf_from_array(traffic_count, src, dst, &iteration_count, &exec_time);

    // int i;
    // printf("final_flow_vector = [");
    // for (i = 0; i < traffic_count - 1; i++)
    //     printf("%f, ", final_flow_vector[i]);
    // printf("%f]\n", final_flow_vector[i]);
    // printf("%d\t %8.6lf (s)\n", iteration_count, exec_time);
    // printf("%d\t %d\n", iteration_count, traffic_count);
}

void get_fct_mmf(unsigned int n_flows, Flow* flows, int h, int *topo, int method_mmf, int method_routing, int type_topo, int base_lr)
{
    assert (type_topo==PL);
    assert (method_routing==PL_ECMP_ROUTING);
    if (method_mmf==PL_TWO_LAYER){
        long long int BW[2];
        for (int i = 0; i < 2; i++)
            BW[i] = topo[i] * ((long long int)base_lr);
        pl_topology_init_two_layer(h, BW);
        pl_routing_init_two_layer();
    }
    else if (method_mmf==PL_ONE_LAYER){
        long long int BW[2];
        for (int i = 0; i < 2; i++)
            BW[i] = topo[i] * ((long long int)base_lr);
        pl_topology_init_one_layer(h, BW);
        pl_routing_init_one_layer();
    }
    else{
        assert(false);
    }

    double t = 0.0;
    unsigned int j = 0;
    unsigned int t_index = 0;
    unsigned int *active_flows_idx = (unsigned int *)malloc(n_flows * sizeof(unsigned int));
    double *t_flows = (double *)malloc((2 * n_flows) * sizeof(double));
    unsigned int *num_flows = (unsigned int *)malloc((2 * n_flows) * sizeof(unsigned int));
    unsigned int *num_flows_enq = (unsigned int *)malloc((n_flows) * sizeof(unsigned int));
    // double lr = 10.0;

    memset(num_flows, 0, 2 * n_flows * sizeof(unsigned int));
    memset(num_flows_enq, 0, n_flows * sizeof(unsigned int));
    // double a_nan = strtod("NaN", NULL);
    double time_to_next_arrival = NAN;
    double time_to_next_completion = NAN;
    unsigned int num_active_flows = 0;
    double sum_weights = 0.0;
    int min_remaining_time_index = -1;

    int *src_active = (int *)malloc(n_flows * sizeof(int));
    int *dst_active = (int *)malloc(n_flows * sizeof(int));

    while (true)
    {
        if (j < n_flows)
        {
            time_to_next_arrival = flows[j].fat - t;
            // printf("time_to_next_arrival:%f\n", time_to_next_arrival);
            assert(time_to_next_arrival >= 0);
        }
        else
        {
            time_to_next_arrival = NAN;
        }
        min_remaining_time_index = -1;
        if (num_active_flows)
        {
            update_rate_mmf(num_active_flows, src_active, dst_active, method_mmf, type_topo);

            time_to_next_completion = INFINITY;
            for (int i = 0; i < num_active_flows; i++)
            {
                unsigned int flow_idx = active_flows_idx[i];
                double remaining_time = flows[flow_idx].remaining_size / final_flow_vector[i];
                if (remaining_time < time_to_next_completion)
                {
                    time_to_next_completion = remaining_time;
                    min_remaining_time_index = i;
                }
            }
        }
        else
        {
            time_to_next_completion = NAN;
        }

        if (num_active_flows > 0 && (j >= n_flows || time_to_next_completion <= time_to_next_arrival))
        {
            // Completion Event
            for (int i = 0; i < num_active_flows; i++)
            {
                unsigned int flow_idx = active_flows_idx[i];
                flows[flow_idx].fct_est += time_to_next_completion;
                flows[flow_idx].remaining_size -= time_to_next_completion * final_flow_vector[i];
            }
            t += time_to_next_completion;
            num_active_flows -= 1;
            assert(min_remaining_time_index != -1);
            active_flows_idx[min_remaining_time_index] = active_flows_idx[num_active_flows];
            src_active[min_remaining_time_index] = src_active[num_active_flows];
            dst_active[min_remaining_time_index] = dst_active[num_active_flows];
        }
        else
        {
            // Arrival Event
            if (j >= n_flows)
            {
                // No more flows left - terminate
                break;
            }
            for (int i = 0; i < num_active_flows; i++)
            {
                unsigned int flow_idx = active_flows_idx[i];
                flows[flow_idx].fct_est += time_to_next_arrival;
                flows[flow_idx].remaining_size -= time_to_next_arrival * final_flow_vector[i];
            }
            t += time_to_next_arrival;
            flows[j].remaining_size = (flows[j].size + ceil(flows[j].size/ MTU) * HEADER_SIZE) * BYTE_TO_BIT;
            flows[j].fct_est = 0.0;
            // active_flows[j].remaining_size = sizes[j] * 8.0;
            active_flows_idx[num_active_flows] = j;
            src_active[num_active_flows] = flows[j].src;
            dst_active[num_active_flows] = flows[j].dst;
            num_active_flows += 1;
            num_flows_enq[j] = num_active_flows;
            j += 1;
        }
        if (method_mmf==PL_TWO_LAYER) {
            pl_reset_topology_two_layer();
        }
        else if (method_mmf==PL_ONE_LAYER) {
            pl_reset_topology_one_layer();
        }
        else{
            assert(false);
        }
        t_flows[t_index] = t;
        num_flows[t_index] = num_active_flows;
        t_index += 1;
        // if (j % 100000 == 0)
        // {
        //     printf("%d/%d simulated in seconds\n", j, n_flows);
        // }
    }

    free(active_flows_idx);
    free(src_active);
    free(dst_active);
}

// Compare function for sorting FlowData based on flow size
int compare_flow_size(const void *a, const void *b) {
    double diff = (*(Flow*)a).size - (*(Flow*)b).size;

    // Use a small epsilon to account for rounding errors
    if (fabs(diff) < 1e-9) {
        return 0;  // Values are considered equal
    }

    return (diff > 0) ? 1 : -1;
}

int compare_flow_sldn(const void *a, const void *b) {
    float diff = (*(Flow*)a).sldn - (*(Flow*)b).sldn;

    // Use a small epsilon to account for rounding errors
    if (fabs(diff) < 1e-9) {
        return 0;  // Values are considered equal
    }

    return (diff > 0) ? 1 : -1;
}
void calculate_and_save_percentiles(Flow* flows, int n_flows, double buckets[], int n_buckets,
                                    double percentiles[], int n_percentiles, float *feat_map,int pos_start,int bucket_thold, float *const_opt) {
    int feat_index = 0;
    // qsort(flows, n_flows, sizeof(Flow), compare_flow_sldn);

    // float *sldn_total = (float *)calloc(n_percentiles, sizeof(float));
    
    // for (int i = 0; i < n_percentiles; ++i) {
    //     sldn_total[i] = 1.0;
    // }

    // int bucket_start=0;
    // int bucket_end=n_flows;
    // for (int k = 0; k < n_percentiles; ++k) {
    //     double flow_idx_thres = bucket_start + ((bucket_end - bucket_start-1) * percentiles[k] / 100);
    //     if (flow_idx_thres == (int)flow_idx_thres) {
    //         // Exact index, no interpolation needed
    //         sldn_total[feat_index++] = flows[(int)flow_idx_thres].sldn;
    //     } else {
    //         // Interpolate between two adjacent values
    //         int lower_index = (int)flow_idx_thres;
    //         int upper_index = lower_index + 1;

    //         // double lower_value = flows[lower_index].sldn;
    //         // double upper_value = flows[upper_index].sldn;

    //         // sldn_total[feat_index++]= flows[upper_index].sldn;
            
    //         double fraction = flow_idx_thres - lower_index;

    //         // sldn_total[feat_index++] =  lower_value + fraction * (upper_value - lower_value);
    //         if (fraction >=0.5){
    //             sldn_total[feat_index++]= flows[upper_index].sldn;
    //         }
    //         else{
    //             sldn_total[feat_index++]= flows[lower_index].sldn;
    //         }
            
    //     }
    // }
    // sldn_total[0]=0.5;
    qsort(flows, n_flows, sizeof(Flow), compare_flow_size);
    // feat_index = 0;
    int* bucket_starts = (int*)malloc(n_buckets * sizeof(int));
    int* bucket_ends = (int*)malloc(n_buckets * sizeof(int));

    memset(bucket_starts, -1, n_buckets * sizeof(int));
    memset(bucket_ends, -1, n_buckets * sizeof(int));

    // Initialize bucket boundaries
    for (int i = 0; i < n_buckets; ++i) {
        // Check if there are flows in this bucket
        if (i == 0 && flows[0].size < buckets[i]) {
            bucket_starts[i] = 0;
            bucket_ends[i] = 0;
        }
        else if (bucket_starts[i]!=-1){
            bucket_ends[i] = bucket_starts[i];
        }

        // Find the end index of the current bucket
        while (bucket_ends[i] < n_flows && flows[bucket_ends[i]].size < buckets[i]) {
            ++bucket_ends[i];
        }

        // Set the start index of the next bucket to be the end index of the current bucket
        if (i < n_buckets - 1 && bucket_ends[i] < n_flows) {
            bucket_starts[i + 1] = bucket_ends[i];
        }
    }
    if (bucket_starts[n_buckets-1]!=-1){
        bucket_ends[n_buckets-1] = n_flows;
    }
    
    for (int i = 0; i < n_buckets; ++i) {
        int bucket_start = bucket_starts[i];
        int bucket_end = bucket_ends[i];

        if (bucket_start == -1 || abs(bucket_end-bucket_start)<bucket_thold) {
            // No flows in this bucket
            // feat_index+=n_percentiles;
            for (int j=0;j<n_percentiles;j++){
                feat_map[pos_start+feat_index++] = const_opt[j];
            }
            // feat_map[pos_start+feat_index-1] = 0;
            continue;
        }
        // printf("Sort Bucket %d: %d\n", i, bucket_end-bucket_start);
        // Resort flows within the current bucket based on completion time
        qsort(&flows[bucket_start], bucket_end - bucket_start, sizeof(Flow), compare_flow_sldn);

        // if(i==0){
        //     for (int j = 0; j < bucket_end-bucket_start; ++j) {
        //         printf("%d:%f ", flows[bucket_start+j].flowId, flows[bucket_start+j].size);
        //     }
        //     printf("\n");
        // }
        // Calculate percentiles for the flows in the current bucket
        for (int k = 0; k < n_percentiles; ++k) {
            double flow_idx_thres = bucket_start + ((bucket_end - bucket_start-1) * percentiles[k] / 100);
            if (flow_idx_thres == (int)flow_idx_thres) {
                // Exact index, no interpolation needed
                feat_map[pos_start+feat_index++] = flows[(int)flow_idx_thres].sldn;
            } else {
                // Interpolate between two adjacent values
                int lower_index = (int)flow_idx_thres;
                int upper_index = lower_index + 1;

                // double lower_value = flows[lower_index].sldn;
                // double upper_value = flows[upper_index].sldn;

                // feat_map[pos_start+feat_index++] = flows[upper_index].sldn;

                double fraction = flow_idx_thres - lower_index;

                // feat_map[pos_start+feat_index++] =  lower_value + fraction * (upper_value - lower_value);
                if (fraction >=0.5){
                    feat_map[pos_start+feat_index++]= flows[upper_index].sldn;
                }
                else{
                    feat_map[pos_start+feat_index++]= flows[lower_index].sldn;
                }
                
            }
        }
        // feat_map[pos_start+feat_index-n_percentiles]=1.0;
        // feat_map[pos_start+feat_index-1]=log(bucket_end - bucket_start);
    }
    // for (int j=0;j<n_percentiles;j++){
    //     feat_map[pos_start+feat_index++] = sldn_total[j];
    // }
    free(bucket_starts);
    free(bucket_ends);
    // free(sldn_total);
}

void write_vec_to_file(const char *filename, float *vec, int vec_size) {
    FILE *file = fopen(filename, "w");
    
    if (file == NULL) {
        // Handle file opening error
        perror("Error opening file");
        return;
    }

    for (int i = 0; i < vec_size; ++i) {
        fprintf(file, "%lf ", vec[i]);
    }

    fclose(file);
}

void write_vecs_to_file(const char *filename, float *vec_1, int vec_size_1, float *vec_2, int vec_size_2) {
    FILE *file = fopen(filename, "w");
    
    if (file == NULL) {
        // Handle file opening error
        perror("Error opening file");
        return;
    }
    for (int i = 0; i < vec_size_1; ++i) {
        fprintf(file, "%lf ", vec_1[i]+1.0);
    }
    fprintf(file, "\n");
    for (int i = 0; i < vec_size_2; ++i) {
        fprintf(file, "%lf ", vec_2[i]);
    }
    fclose(file);
}

int main(int argc, char *argv[]) {
    long start_total=time_in_ms();
    long start=time_in_ms();
    // default parameters
    char *ckpt_path_llama = NULL;  // e.g. out/model.bin
    char *ckpt_path_mlp = NULL;  // e.g. out/model.bin
    char *data_path = NULL; // e.g., "/data1/lichenni/projects/flow_simulation/data_test/shard0_nflows20_nhosts7_lr10Gbps/flows.txt";
    char *data_path_input = NULL;
    char *data_path_output = NULL;
    int n_embd=100;
    int n_hosts=3;
    int bw=1;
    
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat

    int n_size_bucket_input=10;
    int n_size_bucket_output=4;
    int n_percentiles=100;
    double percentiles[100];
    for (int i = 0; i < n_percentiles; i++)
        percentiles[i] = i+1;
    // double percentiles[20] = {10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
    // double percentiles[20] = {1, 10, 25, 40, 55, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100};
    // double percentiles[20] = {1, 25, 40, 55, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 100};

    // double percentiles[30] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 98.2, 98.4, 98.6, 98.8, 99, 99.2,99.4, 99.6, 99.8, 100, 100};

    // double percentiles[30] = {0, 0, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96, 98, 98.2, 98.4, 98.6, 98.8, 99, 99.2, 99.4, 99.6, 99.8, 100, 100};
    // double percentiles[30] = {1, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96, 98, 98.2, 98.4, 98.6, 98.8, 99, 99.2, 99.4, 99.6, 99.8, 99.9, 100, 100};
    // percentiles[n_percentiles-1]=100;

    // spec feature
    int n_param=19;
    float bfsz = 20;
    float fwin = 18000;
    float enable_pfc=1.0;
    float cc=0;
    float param_1=0.0;
    float param_2=0.0;

    int bucket_thold=1;
    double link_to_delay[7]={0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 4) { 
        ckpt_path_llama = argv[1]; 
        ckpt_path_mlp = argv[2];
        data_path = argv[3];
        data_path_input = malloc(strlen(data_path) + strlen("/flows.txt") + 1); // +1 for null terminator
        strcpy(data_path_input, data_path);
        strcat(data_path_input, "/flows.txt");
        data_path_output = malloc(strlen(data_path) + strlen("/fct_mlsys.txt") + 1); // +1 for null terminator
        strcpy(data_path_output, data_path);
        strcat(data_path_output, "/fct_mlsys.txt");
    } else { error_usage(); }
    for (int i = 4; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'b') { bw = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'e') { n_embd = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { n_hosts = atoi(argv[i + 1]); }
        else if (argv[i][1] == 't') { bucket_thold = atoi(argv[i + 1]); }
        
        else if (argv[i][1] == 'f') { bfsz = atof(argv[i + 1]); }
        else if (argv[i][1] == 'k') { fwin = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { enable_pfc = atof(argv[i + 1]); }
        else if (argv[i][1] == 'c') { cc = atof(argv[i + 1]); }
        else if (argv[i][1] == 'x') { param_1 = atof(argv[i + 1]); }
        else if (argv[i][1] == 'y') { param_2 = atof(argv[i + 1]); }
        else { error_usage(); }
    }
    
    double BDP=10.0*MTU;
    double size_bucket_input[9] = {MTU/4.0,MTU/2.0,MTU*3.0/4.0,MTU,BDP/5.0,BDP/2.0,BDP*3.0/4.0,BDP,5.0*BDP};

    float param_list[19]={0,0,0,0, bfsz,fwin/1000.0, 0,0, 0,0,0,0,0,0,0,0,0,0,0};
    if (n_hosts==3){
        param_list[0]=1.0;
        param_list[3]=5.0;
        // link_to_delay[1]=2*DELAY_PROP_BASE;
    }
    else if (n_hosts==5){
        param_list[1]=1.0;
        param_list[3] = 10.0;
        // link_to_delay[1]=2*DELAY_PROP_BASE;
        link_to_delay[2]=1*DELAY_PROP_BASE;
        // link_to_delay[3]=2*DELAY_PROP_BASE;
    }
    else if (n_hosts==7){
        param_list[2]=1.0;
        param_list[3] = 15.0;
        // link_to_delay[1]=2*DELAY_PROP_BASE;
        link_to_delay[2]=1*DELAY_PROP_BASE;
        link_to_delay[3]=2*DELAY_PROP_BASE;
        link_to_delay[4]=1*DELAY_PROP_BASE;
        // link_to_delay[5]=2*DELAY_PROP_BASE;
    }
    if (enable_pfc==1.0){
        param_list[6]=1.0;
    }
    else{
        param_list[7]=1.0;
    };
    if (cc==0){
        param_list[8]=1.0;
        param_list[12]=param_1;
    }
    else if (cc==1){
        param_list[9]=1.0;
        param_list[13]=param_1;
        param_list[14]=param_2;
    }
    else if (cc==2){
        param_list[10]=1.0;
        param_list[15]=param_1;
        param_list[16]=param_2;
    }
    else if (cc==3){
        param_list[11]=1.0;
        param_list[17]=param_1;
        param_list[18]=param_2;
    }
    else{
        printf("cc not supported");
        return 1;
    }
    for (int i =0;i<n_param;i++){
        printf("%f, ", param_list[i]);
    }
    printf("\n");

    // parameter validation/overrides
    int topo[2] = {1, 4};
    float bw_list[2];
    for (int i = 0; i < 2; i++)
        bw_list[i] = (float) topo[i] * bw;
    int src_dst_pair_target[2] = {0,n_hosts-1};
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    // printf("%s, %s, %s, %s, %d, %d, %d, %d\n, ", ckpt_path_llama, ckpt_path_mlp, data_path_input, data_path_output, bw, n_embd, n_hosts, param_list[0]);

    int n_feat_input = n_size_bucket_input*n_percentiles;
    int n_feat_map = (n_feat_input+n_param)*n_hosts;
    int n_feat_context = n_embd;
    int n_feat_concat = n_feat_input+n_param+n_embd;
    int n_feat_output = n_size_bucket_output*n_percentiles;

    // load model and data
    FILE *file = fopen(data_path_input, "r");
    if (!file) { fprintf(stderr, "couldn't load %s\n", data_path_input); exit(EXIT_FAILURE); }
    unsigned int n_flows;
    if (fscanf(file, "%d", &n_flows) != 1) {
        fprintf(stderr, "Error reading the number of lines.\n");
        fclose(file);
        return 1;
    }
    // printf("n_flows: %d\n",  n_flows);
    // Create a vector to store Flow structures
    Flow *flows = malloc(n_flows * sizeof(Flow));
    uint32_t s_port, d_port;
    // Read and save each line into the vector
    for (int i = 0; i < n_flows; ++i) {
        if (fscanf(file, "%d %d %d %d %d %lf %lf",
                   &flows[i].flowId, &flows[i].src, &flows[i].dst,
                   &s_port, &d_port, &flows[i].size,
                   &flows[i].fat) != 7) {
            fprintf(stderr, "Error reading line %d.\n", i + 2);
            free(flows);
            fclose(file);
            return 1;
        }
        else{
            flows[i].idx = i;
            flows[i].fat = flows[i].fat *1e9;
            flows[i].bw_bottleneck=bw_list[0];
            // if (flows[i].src!=0 && flows[i].dst!=n_hosts-1){
            //     flows[i].bw_bottleneck = bw_list[1];
            // }
            // else{
            //     flows[i].bw_bottleneck = bw_list[0];
            // }
        }
    }
    fclose(file);

    // Check the input
    // for (int i = 0; i < n_flows; ++i) {
    //     printf("Flow %d: %d %d %d %lf %lf\n",
    //            flows[i].idx, flows[i].flowId, flows[i].src, flows[i].dst,
    //            flows[i].size, flows[i].fat);
    // }
    // printf("Flow %d: %d %d %d %lf %lf\n",
    //            flows[n_flows-1].idx, flows[n_flows-1].flowId, flows[n_flows-1].src, flows[n_flows-1].dst,
    //            flows[n_flows-1].size, flows[n_flows-1].fat);
    long end = time_in_ms();
    printf( "init: %fs\n",(double)(end-start)/1000);
    start = time_in_ms();

    // run max-min fair rate allocation
    get_fct_mmf(n_flows, flows, n_hosts, topo, PL_ONE_LAYER, PL_ECMP_ROUTING, PL, bw);

    // Check the output
    // for (int i = 0; i < n_flows; ++i) {
    //     printf("Flow %d: %lf %lf %lf\n",
    //            flows[i].flowId, flows[i].remaining_size,flows[i].fct_est, flows[i].sldn);
    // }

    end = time_in_ms();
    printf( "maxmin-fair: %fs\n",(double)(end-start)/1000);
    start = time_in_ms();

    // load_data
    double base_delay=0.0;
    double pkt_header=0.0;
    double i_fct=0.0;
    int n_link=1;
    bool flow_idx_target=false;
    bool flow_idx_nontarget_spec=false;
    bool bottle_link_per_flow=false;
    for (int i = 0; i < n_flows; ++i) {
        pkt_header=flows[i].size<MTU?flows[i].size:MTU;
        pkt_header=(pkt_header + HEADER_SIZE) * BYTE_TO_BIT;
        n_link=abs(flows[i].src-flows[i].dst);
        flow_idx_target=(flows[i].src==0) && (flows[i].dst==n_hosts-1);
        flow_idx_nontarget_spec=(flows[i].src!=0) && (flows[i].dst!=n_hosts-1);

        if (!flow_idx_target) {
            n_link+=1;
        }
        if (flow_idx_nontarget_spec) {
            n_link+=1;
        }

        base_delay=pkt_header / flows[i].bw_bottleneck/4.0*(n_link-2)+DELAY_PROP_BASE * n_link+link_to_delay[flows[i].src]+link_to_delay[flows[i].dst];

        if (flow_idx_target){
            base_delay+= pkt_header / flows[i].bw_bottleneck;
        }

        if (flow_idx_nontarget_spec){
            base_delay-= pkt_header / flows[i].bw_bottleneck;
        }

        i_fct=(flows[i].size + ceil(flows[i].size/ MTU) * HEADER_SIZE) * BYTE_TO_BIT/flows[i].bw_bottleneck;
        flows[i].sldn = (flows[i].fct_est + base_delay) / (i_fct+base_delay);
        assert (flows[i].sldn>=1.0);
    }
    
    // load_model
    
    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, ckpt_path_llama);

    // load MLP
    ConfigMLP config_mlp;
    ModelWeightsMLP weights_mlp;
    RunStateMLP state_mlp;
    build_mlp(ckpt_path_mlp, &config_mlp, &weights_mlp, &state_mlp);
    printf("const_opt: %f, %f\n", weights_mlp.const_opt[0], weights_mlp.const_opt[n_percentiles-1]);

    // printf("\nconfig: %d,%d,%d,%d\n", config_mlp.input_dim, config_mlp.hidden_dim_1, config_mlp.hidden_dim_2, config_mlp.output_dim);
    // printf("MLP loaded!\n");

    end = time_in_ms();
    printf( "build-model: %fs\n",(double)(end-start)/1000);
    start = time_in_ms();

    // printf("Flow %d: %lf %lf\n",flows[n_flows-1].flowId,flows[n_flows-1].fct_est, flows[n_flows-1].sldn);

    float *feat_map = (float *)malloc(n_feat_map * sizeof(float));
    float *feat_concat = (float *)calloc(n_feat_concat, sizeof(float));
    
    for (int i = 0; i < n_feat_map; ++i) {
        feat_map[i] = 0.0;
    }

    int *n_flows_per_link=(int *)malloc(n_hosts * sizeof(int));
    for (int i = 0; i < n_hosts; ++i) {
        n_flows_per_link[i] = 0;
    }

    for (int i = 0; i < n_flows; i++) {
        if (flows[i].src == src_dst_pair_target[0] && flows[i].dst == src_dst_pair_target[1]) {
            n_flows_per_link[n_hosts-1]++;
        }
        else{
            for (int j=flows[i].src;j<flows[i].dst;j++){
                n_flows_per_link[j]++;
            }
        }
    }
    // for (int i = 0; i < n_hosts; i++) {
    //     printf("n_flows_per_link %d: %d\n", i, n_flows_per_link[i]);
    // }
    int feat_pos=0;
    int n_flows_fg=n_flows_per_link[n_hosts-1];
    int* flow_ids_bg;
    int n_flows_bg=n_flows-n_flows_fg;
    flow_ids_bg = (int *)malloc(n_flows_bg * sizeof(int));
    Flow* flows_fg;

    if (n_flows_fg>0){
        flows_fg = (Flow *)malloc(n_flows_fg * sizeof(Flow));
    }
    int index_fg = 0;
    int index_bg = 0;

    for (int i = 0; i < n_flows; ++i) {
        if (flows[i].src == src_dst_pair_target[0] && flows[i].dst == src_dst_pair_target[1]) {
            flows_fg[index_fg++] = flows[i];
        }
        else{
            flow_ids_bg[index_bg++] = i;
        }
    }
    if (n_flows_fg>0){
        // calcuate the feature map
        calculate_and_save_percentiles(flows_fg, n_flows_fg, size_bucket_input, n_size_bucket_input, percentiles, n_percentiles, feat_map,feat_pos,bucket_thold,weights_mlp.const_opt);
        // for (int i = 0; i < n_feat_input; ++i) {
        //     printf("feat_input %d: %lf\n", i, feat_input[i]);
        // }
        free(flows_fg);
    }
    // for(int i=0; i<n_feat_input; i++){
    //     feat_map[feat_pos+i]-=1.0;
    // }
    // param_list[n_param-1]=BDP_path/MTU;
    for(int i=0; i<n_param; i++){
        feat_map[feat_pos+n_feat_input+i]=param_list[i];
    }
    // printf("feat-input-%d: %d, %lf, %lf, %lf\n", feat_pos, n_flows_fg, feat_map[feat_pos], feat_map[feat_pos+n_feat_input-1], feat_map[feat_pos+n_feat_input]);

    int flow_id_bg;
    for (int linkid_idx = 0; linkid_idx < n_hosts-1; ++linkid_idx) {
        n_flows_fg=n_flows_per_link[linkid_idx];
        index_fg = 0;
        feat_pos=(linkid_idx+1)*(n_feat_input+n_param);
        if (n_flows_fg>0){
            flows_fg = (Flow *)malloc(n_flows_fg * sizeof(Flow));
            // Count the number of flows satisfying the condition
            for (int i = 0; i < n_flows_bg; ++i) {
                flow_id_bg=flow_ids_bg[i];
                if (flows[flow_id_bg].src <= linkid_idx && flows[flow_id_bg].dst > linkid_idx) {
                    flows_fg[index_fg++] = flows[flow_id_bg];
                }
            }
            // calcuate the feature map
            calculate_and_save_percentiles(flows_fg, n_flows_fg, size_bucket_input, n_size_bucket_input, percentiles, n_percentiles, feat_map,feat_pos,bucket_thold,weights_mlp.const_opt);
            // for (int i = 0; i < n_feat_input; ++i) {
            //     printf("feat_input %d: %lf\n", i, feat_input[i]);
            // }
            free(flows_fg);
        }
        // for(int i=0; i<n_feat_input; i++){
        //     feat_map[feat_pos+i]-=1.0;
        // }
        for(int i=0; i<n_param; i++){
            feat_map[feat_pos+n_feat_input+i]=param_list[i];
        }
        // printf("feat_input-%d: %d, %lf, %lf, %lf\n", linkid_idx+1,n_flows_fg, feat_map[feat_pos], feat_map[feat_pos+n_feat_input-1], feat_map[feat_pos+n_feat_input]);
    }

    end = time_in_ms();
    printf( "feat-map: %fs\n",(double)(end-start)/1000);
    start = time_in_ms();

    // write_vec_to_file("./feat_map_c.txt", feat_map, n_feat_map);

    // run!
    // feat_map: ((n_feat_input+n_param)*n_hosts,1)
    if (strcmp(mode, "generate") == 0) {
        for (int i = 0; i < n_feat_input+n_param; ++i) {
            feat_concat[i] = feat_map[i];
        }
        generate(&transformer, feat_map, n_hosts,n_feat_input+n_param,n_feat_context,feat_concat);
        // printf("feat_mlp %lf, %lf, %lf, %lf\n", feat_concat[0], feat_concat[n_feat_input], feat_concat[n_feat_input+n_param], feat_concat[n_feat_input+n_param+n_feat_context-1]);

        end = time_in_ms();
        printf( "transformer: %fs\n",(double)(end-start)/1000);
        start = time_in_ms();

        //run mlp
        forward_MLP(feat_concat, &config_mlp, &weights_mlp, &state_mlp);

        float *feat_output = state_mlp.logits;

        end = time_in_ms();
        printf( "mlp: %fs\n",(double)(end-start)/1000);
        start = time_in_ms();

        printf("feat_output: %lf, %lf\n", feat_output[0], feat_output[n_feat_output-1]);
        write_vecs_to_file(data_path_output, feat_output, n_feat_output,feat_concat,n_feat_concat);

    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }
    end = time_in_ms();
    printf( "time: %fs\n",(double)(end-start_total)/1000);
    // memory and file handles cleanup
    free(flows);
    free(feat_map);
    free(feat_concat);
    free(n_flows_per_link);
    free(flow_ids_bg);
    free_transformer(&transformer);
    free_run_state_mlp(&state_mlp);
    return 0;
}
#endif

