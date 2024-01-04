//
// Created by Stefan on 1/3/2024.
// Inspired by the stable-diffusion.cpp library. Licensed under the MIT license to Leejet.
//

#pragma once

#include "util/std.hpp"

#include <ggml/ggml-alloc.h>
#include <ggml/ggml-backend.h>
#include <ggml/ggml.h>

#ifdef SD_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef SD_USE_METAL
#include "ggml-metal.h"
#endif

#include <model.h>
#include <rng.h>
#include <rng_philox.h>
#include <stable-diffusion.h>
#include <util.h>

#define EPS 1e-05f

#define UNET_GRAPH_SIZE 10240
#define LORA_GRAPH_SIZE 10240

#define TIMESTEPS 1000

static const char* model_version_to_str[] = {
    "1.x",
    "2.x",
    "XL",
};

static const char* sampling_methods_str[] = {
    "Euler A",
    "Euler",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "LCM",
};

/*================================================== Helper Functions ================================================*/

std::string sd_get_system_info();

static void ggml_log_callback_default(ggml_log_level level, const char* text, void* user_data);

void ggml_tensor_set_f32_randn(struct ggml_tensor* tensor, std::shared_ptr<RNG> rng);

void pretty_progress(int step, int steps, float time);

// set tensor[i, j, k, l]
// set tensor[l]
// set tensor[k, l]
// set tensor[j, k, l]
void ggml_tensor_set_f32(struct ggml_tensor* tensor, float value, int l, int k = 0, int j = 0, int i = 0);

float ggml_tensor_get_f32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0);

ggml_fp16_t ggml_tensor_get_f16(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0);

void print_ggml_tensor(struct ggml_tensor* tensor, bool shape_only = false);

ggml_tensor* load_tensor_from_file(ggml_context* ctx, const std::string& file_path);

void sd_fread(void* ptr, size_t size, size_t count, FILE* stream);

void copy_ggml_tensor(struct ggml_tensor* dst, struct ggml_tensor* src);

void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start = 0.00085f,
                              float linear_end   = 0.0120,
                              int timesteps      = TIMESTEPS);

// Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
void set_timestep_embedding(struct ggml_tensor* timesteps, struct ggml_tensor* embedding, int dim, int max_period = 10000);

struct ggml_tensor* new_timestep_embedding(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* timesteps, int dim, int max_period = 10000);

// SPECIAL OPERATIONS WITH TENSORS

uint8_t* sd_tensor_to_image(struct ggml_tensor* input);

void sd_image_to_tensor(const uint8_t* image_data,
                        struct ggml_tensor* output);

void ggml_split_tensor_2d(struct ggml_tensor* input,
                          struct ggml_tensor* output,
                          int x,
                          int y);

void ggml_merge_tensor_2d(struct ggml_tensor* input,
                          struct ggml_tensor* output,
                          int x,
                          int y,
                          int overlap);

float ggml_tensor_mean(struct ggml_tensor* src);

// a = a+b
void ggml_tensor_add(struct ggml_tensor* a, struct ggml_tensor* b);

void ggml_tensor_scale(struct ggml_tensor* src, float scale);

void ggml_tensor_clamp(struct ggml_tensor* src, float min, float max);

// convert values from [0, 1] to [-1, 1]
void ggml_tensor_scale_input(struct ggml_tensor* src);

// convert values from [-1, 1] to [0, 1]
void ggml_tensor_scale_output(struct ggml_tensor* src);

typedef std::function<void(ggml_tensor*, ggml_tensor*, bool)> on_tile_process;

// Tiling
void sd_tiling(ggml_tensor* input, ggml_tensor* output, const int scale, const int tile_size, const float tile_overlap_factor, on_tile_process on_processing);

struct ggml_tensor* ggml_group_norm_32(struct ggml_context* ctx,
                                       struct ggml_tensor* a);

struct ggml_tensor* ggml_nn_linear(struct ggml_context* ctx,
                                   struct ggml_tensor* x,
                                   struct ggml_tensor* w,
                                   struct ggml_tensor* b);

// w: [OC，IC, KH, KW]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N, OC, OH, OW]
struct ggml_tensor* ggml_nn_conv_2d(struct ggml_context* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* w,
                                    struct ggml_tensor* b,
                                    int s0 = 1,
                                    int s1 = 1,
                                    int p0 = 0,
                                    int p1 = 0,
                                    int d0 = 1,
                                    int d1 = 1);

struct ggml_tensor* ggml_nn_layer_norm(struct ggml_context* ctx,
                                       struct ggml_tensor* x,
                                       struct ggml_tensor* w,
                                       struct ggml_tensor* b,
                                       float eps = EPS);

struct ggml_tensor* ggml_nn_group_norm(struct ggml_context* ctx,
                                       struct ggml_tensor* x,
                                       struct ggml_tensor* w,
                                       struct ggml_tensor* b,
                                       int num_groups = 32);

std::pair<std::unordered_map<std::string, float>, std::string> extract_and_remove_lora(std::string text);

void ggml_backend_tensor_get_and_sync(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);

/*================================================== CLIPTokenizer ===================================================*/

const std::string UNK_TOKEN = "<|endoftext|>";
const std::string BOS_TOKEN = "<|startoftext|>";
const std::string EOS_TOKEN = "<|endoftext|>";
const std::string PAD_TOEKN = "<|endoftext|>";

const int UNK_TOKEN_ID = 49407;
const int BOS_TOKEN_ID = 49406;
const int EOS_TOKEN_ID = 49407;
const int PAD_TOKEN_ID = 49407;

std::vector<std::pair<int, std::u32string>> bytes_to_unicode();

// Ref: https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
class CLIPTokenizer {
 private:
  SDVersion version = VERSION_1_x;
  std::map<int, std::u32string> byte_encoder;
  std::map<std::u32string, int> encoder;
  std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks;
  std::regex pat;

  static std::string strip(const std::string& str);

  static std::string whitespace_clean(std::string text);

  static std::set<std::pair<std::u32string, std::u32string>> get_pairs(const std::vector<std::u32string>& subwords);

 public:
  CLIPTokenizer(SDVersion version = VERSION_1_x);

  void load_from_merges(const std::string& merges_utf8_str);

  std::u32string bpe(const std::u32string& token);

  std::vector<int> tokenize(std::string text, size_t max_length = 0, bool padding = false);

  std::vector<int> encode(std::string text);
};


std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text);

/*================================================ FrozenCLIPEmbedder ================================================*/

struct ResidualAttentionBlock {
  int32_t n_head;
  int32_t d_model;
  int32_t hidden_size;  // n_head * d_model
  int32_t intermediate_size;

  // attention
  struct ggml_tensor* q_w;  // [hidden_size, hidden_size]
  struct ggml_tensor* q_b;  // [hidden_size, ]
  struct ggml_tensor* k_w;  // [hidden_size, hidden_size]
  struct ggml_tensor* k_b;  // [hidden_size, ]
  struct ggml_tensor* v_w;  // [hidden_size, hidden_size]
  struct ggml_tensor* v_b;  // [hidden_size, ]

  struct ggml_tensor* out_w;  // [hidden_size, hidden_size]
  struct ggml_tensor* out_b;  // [hidden_size, ]

  // layer norm 1
  struct ggml_tensor* ln1_w;  // [hidden_size, ]
  struct ggml_tensor* ln1_b;  // [hidden_size, ]

  // mlp
  struct ggml_tensor* fc1_w;  // [intermediate_size, hidden_size]
  struct ggml_tensor* fc1_b;  // [intermediate_size, ]

  struct ggml_tensor* fc2_w;  // [hidden_size, intermediate_size]
  struct ggml_tensor* fc2_b;  // [hidden_size, ]

  // layer norm 2
  struct ggml_tensor* ln2_w;  // [hidden_size, ]
  struct ggml_tensor* ln2_b;  // [hidden_size, ]

  struct ggml_tensor* attn_scale;  // [hidden_size, ]

  size_t calculate_mem_size(ggml_type wtype);

  void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype);

  void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix);

  struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x);
};

// OPENAI_CLIP_VIT_L_14: https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
// OPEN_CLIP_VIT_H_14: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/config.json
// OPEN_CLIP_VIT_BIGG_14: https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/blob/main/config.json (CLIPTextModelWithProjection)
// SDXL CLIPModel
// CLIPTextModelWithProjection seems optional

enum CLIPVersion {
  OPENAI_CLIP_VIT_L_14,   // SD 1.x and SDXL
  OPEN_CLIP_VIT_H_14,     // SD 2.x
  OPEN_CLIP_VIT_BIGG_14,  // SDXL
};

struct CLIPTextModel {
  CLIPVersion version = OPENAI_CLIP_VIT_L_14;
  // network hparams
  int32_t vocab_size              = 49408;
  int32_t max_position_embeddings = 77;
  int32_t hidden_size             = 768;   // 1024 for OPEN_CLIP_VIT_H_14
  int32_t intermediate_size       = 3072;  // 4096 for OPEN_CLIP_VIT_H_14
  int32_t n_head                  = 12;    // num_attention_heads, 16 for OPEN_CLIP_VIT_H_14
  int32_t num_hidden_layers       = 12;    // 24 for OPEN_CLIP_VIT_H_14
  int32_t layer_idx               = 11;
  int32_t projection_dim          = 1280;  // only for OPEN_CLIP_VIT_BIGG_14
  bool with_final_ln              = true;

  // embeddings
  struct ggml_tensor* position_ids;
  struct ggml_tensor* token_embed_weight;
  struct ggml_tensor* position_embed_weight;

  // transformer
  std::vector<ResidualAttentionBlock> resblocks;
  struct ggml_tensor* final_ln_w;
  struct ggml_tensor* final_ln_b;

  struct ggml_tensor* text_projection;

  CLIPTextModel(CLIPVersion version = OPENAI_CLIP_VIT_L_14,
                int clip_skip       = 1,
                bool with_final_ln  = true);

  void set_resblocks_hp_params();

  size_t calculate_mem_size(ggml_type wtype);

  struct ggml_tensor* forward(struct ggml_context* ctx0, struct ggml_tensor* input_ids, uint32_t max_token_idx = 0, bool return_pooled = false);

  void alloc_params(ggml_context* ctx, ggml_backend_t backend, ggml_type wtype, ggml_allocr* alloc);
};

// ldm.modules.encoders.modules.FrozenCLIPEmbedder
struct FrozenCLIPEmbedder {
  CLIPTokenizer tokenizer;
  CLIPTextModel text_model;

  struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_allocr* allocr, const std::string& prompt);
};

// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/sd_hijack_clip.py#L283
struct FrozenCLIPEmbedderWithCustomWords {
  SDVersion version = VERSION_1_x;
  CLIPTokenizer tokenizer;
  CLIPTextModel text_model;
  CLIPTextModel text_model2;

  // context and memory buffers
  struct ggml_context* ctx  = NULL;
  ggml_backend_buffer_t params_buffer = NULL;
  ggml_backend_buffer_t compute_buffer  = NULL;;  // for compute
  struct ggml_allocr* compute_alloc = NULL;
  size_t compute_memory_buffer_size = -1;

  size_t memory_buffer_size = 0;
  ggml_type wtype;
  ggml_backend_t backend           = NULL;
  ggml_tensor* hidden_state_output = NULL;
  ggml_tensor* pooled_output       = NULL;

  FrozenCLIPEmbedderWithCustomWords(SDVersion version = VERSION_1_x, int clip_skip = -1);

  void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix);

  struct ggml_tensor* forward(struct ggml_context* ctx0, struct ggml_tensor* input_ids, struct ggml_tensor* input_ids2, uint32_t max_token_idx = 0, bool return_pooled = false);

  std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                           bool padding = false);

  std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                           size_t max_length = 0,
                                                           bool padding      = false);

  bool initialize(ggml_backend_t backend_, ggml_type wtype_);

  void destroy();

  void alloc_params();

  struct ggml_cgraph* build_graph(struct ggml_allocr* allocr, std::vector<int> tokens, bool return_pooled = false);

  void begin(ggml_context* work_ctx, int max_tokens);

  std::pair<struct ggml_tensor*, struct ggml_tensor*> compute(const int n_threads, std::vector<int> tokens);

  void end();
};

/*==================================================== UnetModel =====================================================*/

struct ResBlock {
  // network hparams
  int channels;      // model_channels * (1, 1, 1, 2, 2, 4, 4, 4)
  int emb_channels;  // time_embed_dim
  int out_channels;  // mult * model_channels

  // network params
  // in_layers
  struct ggml_tensor* in_layer_0_w;  // [channels, ]
  struct ggml_tensor* in_layer_0_b;  // [channels, ]
  // in_layer_1 is nn.SILU()
  struct ggml_tensor* in_layer_2_w;  // [out_channels, channels, 3, 3]
  struct ggml_tensor* in_layer_2_b;  // [out_channels, ]

  // emb_layers
  // emb_layer_0 is nn.SILU()
  struct ggml_tensor* emb_layer_1_w;  // [out_channels, emb_channels]
  struct ggml_tensor* emb_layer_1_b;  // [out_channels, ]

  // out_layers
  struct ggml_tensor* out_layer_0_w;  // [out_channels, ]
  struct ggml_tensor* out_layer_0_b;  // [out_channels, ]
  // out_layer_1 is nn.SILU()
  // out_layer_2 is nn.Dropout(), p = 0 for inference
  struct ggml_tensor* out_layer_3_w;  // [out_channels, out_channels, 3, 3]
  struct ggml_tensor* out_layer_3_b;  // [out_channels, ]

  // skip connection, only if out_channels != channels
  struct ggml_tensor* skip_w;  // [out_channels, channels, 1, 1]
  struct ggml_tensor* skip_b;  // [out_channels, ]

  size_t calculate_mem_size(ggml_type wtype);

  void init_params(struct ggml_context* ctx, ggml_type wtype);

  void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix);

  struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* emb);
};

struct SpatialTransformer {
  int in_channels;        // mult * model_channels
  int n_head;             // num_heads
  int d_head;             // in_channels // n_heads
  int depth       = 1;    // 1
  int context_dim = 768;  // hidden_size, 1024 for VERSION_2_x

  // group norm
  struct ggml_tensor* norm_w;  // [in_channels,]
  struct ggml_tensor* norm_b;  // [in_channels,]

  // proj_in
  struct ggml_tensor* proj_in_w;  // [in_channels, in_channels, 1, 1]
  struct ggml_tensor* proj_in_b;  // [in_channels,]

  // transformer
  struct Transformer {
    // layer norm 1
    struct ggml_tensor* norm1_w;  // [in_channels, ]
    struct ggml_tensor* norm1_b;  // [in_channels, ]

    // attn1
    struct ggml_tensor* attn1_q_w;  // [in_channels, in_channels]
    struct ggml_tensor* attn1_k_w;  // [in_channels, in_channels]
    struct ggml_tensor* attn1_v_w;  // [in_channels, in_channels]

    struct ggml_tensor* attn1_out_w;  // [in_channels, in_channels]
    struct ggml_tensor* attn1_out_b;  // [in_channels, ]

    // layer norm 2
    struct ggml_tensor* norm2_w;  // [in_channels, ]
    struct ggml_tensor* norm2_b;  // [in_channels, ]

    // attn2
    struct ggml_tensor* attn2_q_w;  // [in_channels, in_channels]
    struct ggml_tensor* attn2_k_w;  // [in_channels, context_dim]
    struct ggml_tensor* attn2_v_w;  // [in_channels, context_dim]

    struct ggml_tensor* attn2_out_w;  // [in_channels, in_channels]
    struct ggml_tensor* attn2_out_b;  // [in_channels, ]

    // layer norm 3
    struct ggml_tensor* norm3_w;  // [in_channels, ]
    struct ggml_tensor* norm3_b;  // [in_channels, ]

    // ff
    struct ggml_tensor* ff_0_proj_w;  // [in_channels * 4 * 2, in_channels]
    struct ggml_tensor* ff_0_proj_b;  // [in_channels * 4 * 2]

    struct ggml_tensor* ff_2_w;  // [in_channels, in_channels * 4]
    struct ggml_tensor* ff_2_b;  // [in_channels,]
  };

  std::vector<Transformer> transformers;

  struct ggml_tensor* attn_scale;

  // proj_out
  struct ggml_tensor* proj_out_w;  // [in_channels, in_channels, 1, 1]
  struct ggml_tensor* proj_out_b;  // [in_channels,]

  SpatialTransformer(int depth = 1);

  size_t get_num_tensors();

  size_t calculate_mem_size(ggml_type wtype);

  void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype);

  void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix);

  struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* context);
};

struct DownSample {
  // hparams
  int channels;
  int out_channels;

  // conv2d params
  struct ggml_tensor* op_w;  // [out_channels, channels, 3, 3]
  struct ggml_tensor* op_b;  // [out_channels,]

  bool vae_downsample = false;

  size_t calculate_mem_size(ggml_type wtype);

  void init_params(struct ggml_context* ctx, ggml_type wtype);

  void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix);

  struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x);
};

struct UpSample {
  // hparams
  int channels;
  int out_channels;

  // conv2d params
  struct ggml_tensor* conv_w;  // [out_channels, channels, 3, 3]
  struct ggml_tensor* conv_b;  // [out_channels,]

  size_t calculate_mem_size(ggml_type wtype);

  void init_params(struct ggml_context* ctx, ggml_type wtype);

  void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix);

  struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x);
};

// ldm.modules.diffusionmodules.openaimodel.UNetModel
struct UNetModel {
  SDVersion version = VERSION_1_x;
  // network hparams
  int in_channels                        = 4;
  int model_channels                     = 320;
  int out_channels                       = 4;
  int num_res_blocks                     = 2;
  std::vector<int> attention_resolutions = {4, 2, 1};
  std::vector<int> channel_mult          = {1, 2, 4, 4};
  std::vector<int> transformer_depth     = {1, 1, 1, 1};
  int time_embed_dim                     = 1280;  // model_channels*4
  int num_heads                          = 8;
  int num_head_channels                  = -1;    // channels // num_heads
  int context_dim                        = 768;   // 1024 for VERSION_2_x, 2048 for VERSION_XL
  int adm_in_channels                    = 2816;  // only for VERSION_XL

  // network params
  struct ggml_tensor* time_embed_0_w;  // [time_embed_dim, model_channels]
  struct ggml_tensor* time_embed_0_b;  // [time_embed_dim, ]
  // time_embed_1 is nn.SILU()
  struct ggml_tensor* time_embed_2_w;  // [time_embed_dim, time_embed_dim]
  struct ggml_tensor* time_embed_2_b;  // [time_embed_dim, ]

  struct ggml_tensor* label_embed_0_w;  // [time_embed_dim, adm_in_channels]
  struct ggml_tensor* label_embed_0_b;  // [time_embed_dim, ]
  // label_embed_1 is nn.SILU()
  struct ggml_tensor* label_embed_2_w;  // [time_embed_dim, time_embed_dim]
  struct ggml_tensor* label_embed_2_b;  // [time_embed_dim, ]

  struct ggml_tensor* input_block_0_w;  // [model_channels, in_channels, 3, 3]
  struct ggml_tensor* input_block_0_b;  // [model_channels, ]

  // input_blocks
  ResBlock input_res_blocks[4][2];
  SpatialTransformer input_transformers[3][2];
  DownSample input_down_samples[3];

  // middle_block
  ResBlock middle_block_0;
  SpatialTransformer middle_block_1;
  ResBlock middle_block_2;

  // output_blocks
  ResBlock output_res_blocks[4][3];
  SpatialTransformer output_transformers[3][3];
  UpSample output_up_samples[3];

  // out
  // group norm 32
  struct ggml_tensor* out_0_w;  // [model_channels, ]
  struct ggml_tensor* out_0_b;  // [model_channels, ]
  // out 1 is nn.SILU()
  struct ggml_tensor* out_2_w;  // [out_channels, model_channels, 3, 3]
  struct ggml_tensor* out_2_b;  // [out_channels, ]

  struct ggml_context* ctx;
  ggml_backend_buffer_t params_buffer;
  ggml_backend_buffer_t compute_buffer;  // for compute
  struct ggml_allocr* compute_alloc = NULL;
  size_t compute_memory_buffer_size = -1;

  size_t memory_buffer_size = 0;
  ggml_type wtype;
  ggml_backend_t backend = NULL;

  UNetModel(SDVersion version = VERSION_1_x);

  size_t calculate_mem_size();

  int get_num_tensors() ;

  bool initialize(ggml_backend_t backend_, ggml_type wtype_);

  void alloc_params();

  void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix);

  struct ggml_tensor* forward(struct ggml_context* ctx0,
                              struct ggml_tensor* x,
                              struct ggml_tensor* timesteps,
                              struct ggml_tensor* context,
                              struct ggml_tensor* t_emb = NULL,
                              struct ggml_tensor* y     = NULL);

  struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                  struct ggml_tensor* timesteps,
                                  struct ggml_tensor* context,
                                  struct ggml_tensor* t_emb = NULL,
                                  struct ggml_tensor* y     = NULL);

  void begin(struct ggml_tensor* x,
             struct ggml_tensor* context,
             struct ggml_tensor* t_emb = NULL,
             struct ggml_tensor* y     = NULL);

  void compute(struct ggml_tensor* work_latent,
               int n_threads,
               struct ggml_tensor* x,
               struct ggml_tensor* timesteps,
               struct ggml_tensor* context,
               struct ggml_tensor* t_emb = NULL,
               struct ggml_tensor* y     = NULL);

  void end();
};

/*================================================== AutoEncoderKL ===================================================*/

struct ResnetBlock {
  // network hparams
  int in_channels;
  int out_channels;

  // network params
  struct ggml_tensor* norm1_w;  // [in_channels, ]
  struct ggml_tensor* norm1_b;  // [in_channels, ]

  struct ggml_tensor* conv1_w;  // [out_channels, in_channels, 3, 3]
  struct ggml_tensor* conv1_b;  // [out_channels, ]

  struct ggml_tensor* norm2_w;  // [out_channels, ]
  struct ggml_tensor* norm2_b;  // [out_channels, ]

  struct ggml_tensor* conv2_w;  // [out_channels, out_channels, 3, 3]
  struct ggml_tensor* conv2_b;  // [out_channels, ]

  // nin_shortcut, only if out_channels != in_channels
  struct ggml_tensor* nin_shortcut_w;  // [out_channels, in_channels, 1, 1]
  struct ggml_tensor* nin_shortcut_b;  // [out_channels, ]

  size_t calculate_mem_size(ggml_type wtype);

  void init_params(struct ggml_context* ctx, ggml_type wtype);

  void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix);

  struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* z);
};

struct AttnBlock {
  int in_channels;  // mult * model_channels

  // group norm
  struct ggml_tensor* norm_w;  // [in_channels,]
  struct ggml_tensor* norm_b;  // [in_channels,]

  // q/k/v
  struct ggml_tensor* q_w;  // [in_channels, in_channels, 1, 1]
  struct ggml_tensor* q_b;  // [in_channels,]
  struct ggml_tensor* k_w;  // [in_channels, in_channels, 1, 1]
  struct ggml_tensor* k_b;  // [in_channels,]
  struct ggml_tensor* v_w;  // [in_channels, in_channels, 1, 1]
  struct ggml_tensor* v_b;  // [in_channels,]

  // proj_out
  struct ggml_tensor* proj_out_w;  // [in_channels, in_channels, 1, 1]
  struct ggml_tensor* proj_out_b;  // [in_channels,]

  struct ggml_tensor* attn_scale;

  size_t calculate_mem_size(ggml_type wtype);

  void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype);

  void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix);

  struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x);
};

// ldm.modules.diffusionmodules.model.Encoder
struct Encoder {
  int embed_dim      = 4;
  int ch             = 128;
  int z_channels     = 4;
  int in_channels    = 3;
  int num_res_blocks = 2;
  int ch_mult[4]     = {1, 2, 4, 4};

  struct ggml_tensor* conv_in_w;  // [ch, in_channels, 3, 3]
  struct ggml_tensor* conv_in_b;  // [ch, ]

  ResnetBlock down_blocks[4][2];
  DownSample down_samples[3];

  struct
  {
    ResnetBlock block_1;
    AttnBlock attn_1;
    ResnetBlock block_2;
  } mid;

  // block_in = ch * ch_mult[len_mults - 1]
  struct ggml_tensor* norm_out_w;  // [block_in, ]
  struct ggml_tensor* norm_out_b;  // [block_in, ]

  struct ggml_tensor* conv_out_w;  // [embed_dim*2, block_in, 3, 3]
  struct ggml_tensor* conv_out_b;  // [embed_dim*2, ]

  Encoder();

  size_t get_num_tensors();

  size_t calculate_mem_size(ggml_type wtype);

  void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype);

  void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix);

  struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x);
};

// ldm.modules.diffusionmodules.model.Decoder
struct Decoder {
  int embed_dim      = 4;
  int ch             = 128;
  int z_channels     = 4;
  int out_ch         = 3;
  int num_res_blocks = 2;
  int ch_mult[4]     = {1, 2, 4, 4};

  // block_in = ch *  ch_mult[-1], 512
  struct ggml_tensor* conv_in_w;  // [block_in, z_channels, 3, 3]
  struct ggml_tensor* conv_in_b;  // [block_in, ]

  struct
  {
    ResnetBlock block_1;
    AttnBlock attn_1;
    ResnetBlock block_2;
  } mid;

  ResnetBlock up_blocks[4][3];
  UpSample up_samples[3];

  struct ggml_tensor* norm_out_w;  // [ch *  ch_mult[0], ]
  struct ggml_tensor* norm_out_b;  // [ch *  ch_mult[0], ]

  struct ggml_tensor* conv_out_w;  // [out_ch, ch *  ch_mult[0], 3, 3]
  struct ggml_tensor* conv_out_b;  // [out_ch, ]

  Decoder();

  size_t calculate_mem_size(ggml_type wtype);

  size_t get_num_tensors();

  void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype);

  void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix);

  struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* z);
};

// ldm.models.autoencoder.AutoencoderKL
struct AutoEncoderKL {
  bool decode_only = true;
  int embed_dim    = 4;
  struct
  {
    int z_channels     = 4;
    int resolution     = 256;
    int in_channels    = 3;
    int out_ch         = 3;
    int ch             = 128;
    int ch_mult[4]     = {1, 2, 4, 4};
    int num_res_blocks = 2;
  } dd_config;

  struct ggml_tensor* quant_conv_w;  // [2*embed_dim, 2*z_channels, 1, 1]
  struct ggml_tensor* quant_conv_b;  // [2*embed_dim, ]

  struct ggml_tensor* post_quant_conv_w;  // [z_channels, embed_dim, 1, 1]
  struct ggml_tensor* post_quant_conv_b;  // [z_channels, ]

  Encoder encoder;
  Decoder decoder;

  struct ggml_context* ctx  = NULL;
  ggml_backend_buffer_t params_buffer  = NULL;
  ggml_backend_buffer_t compute_buffer  = NULL;  // for compute
  struct ggml_allocr* compute_alloc = NULL;

  int memory_buffer_size = 0;
  ggml_type wtype;
  ggml_backend_t backend = NULL;

  AutoEncoderKL(bool decode_only = false);

  size_t calculate_mem_size();

  void destroy();

  void alloc_params();

  void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix);

  struct ggml_tensor* decode(struct ggml_context* ctx0, struct ggml_tensor* z);

  struct ggml_tensor* encode(struct ggml_context* ctx0, struct ggml_tensor* x);

  struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph);

  void begin(struct ggml_tensor* x, bool decode);

  void end();
};

/*
    ===================================    TinyAutoEncoder  ===================================
    References:
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_tiny.py
    https://github.com/madebyollin/taesd/blob/main/taesd.py

*/
struct TAEBlock {
  int in_channels;
  int out_channels;

  // conv
  ggml_tensor* conv_0_w;  // [in_channels, out_channels, 3, 3]
  ggml_tensor* conv_0_b;  // [in_channels]
  ggml_tensor* conv_1_w;  // [out_channels, out_channels, 3, 3]
  ggml_tensor* conv_1_b;  // [out_channels]
  ggml_tensor* conv_2_w;  // [out_channels, out_channels, 3, 3]
  ggml_tensor* conv_2_b;  // [out_channels]

  // skip
  ggml_tensor* conv_skip_w;  // [in_channels, out_channels, 1, 1]

  size_t calculate_mem_size();

  int get_num_tensors();

  void init_params(ggml_context* ctx);

  void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix);

  ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x);
};

struct TinyEncoder {
  int in_channels = 3;
  int z_channels  = 4;
  int channels    = 64;
  int num_blocks  = 3;

  // input
  ggml_tensor* conv_input_w;  // [channels, in_channels, 3, 3]
  ggml_tensor* conv_input_b;  // [channels]
  TAEBlock initial_block;

  ggml_tensor* conv_1_w;  // [channels, channels, 3, 3]
  TAEBlock input_blocks[3];

  // middle
  ggml_tensor* conv_2_w;  // [channels, channels, 3, 3]
  TAEBlock middle_blocks[3];

  // output
  ggml_tensor* conv_3_w;  // [channels, channels, 3, 3]
  TAEBlock output_blocks[3];

  // final
  ggml_tensor* conv_final_w;  // [z_channels, channels, 3, 3]
  ggml_tensor* conv_final_b;  // [z_channels]

  TinyEncoder();

  size_t calculate_mem_size();

  int get_num_tensors();

  void init_params(ggml_context* ctx);

  void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix);

  ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x);
};

struct TinyDecoder {
  int z_channels      = 4;
  int channels        = 64;
  int output_channels = 3;
  int num_blocks      = 3;

  // input
  ggml_tensor* conv_input_w;  // [channels, z_channels, 3, 3]
  ggml_tensor* conv_input_b;  // [channels]
  TAEBlock input_blocks[3];
  ggml_tensor* conv_1_w;  // [channels, channels, 3, 3]

  // middle
  TAEBlock middle_blocks[3];
  ggml_tensor* conv_2_w;  // [channels, channels, 3, 3]

  // output
  TAEBlock output_blocks[3];
  ggml_tensor* conv_3_w;  // [channels, channels, 3, 3]

  // final
  TAEBlock final_block;
  ggml_tensor* conv_final_w;  // [output_channels, channels, 3, 3]
  ggml_tensor* conv_final_b;  // [output_channels]

  ggml_tensor* in_scale_1d3;  // [1]
  ggml_tensor* in_scale_3;    // [1]

  TinyDecoder();

  size_t calculate_mem_size();

  int get_num_tensors();

  void init_params(ggml_allocr* alloc, ggml_context* ctx);

  void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix);

  ggml_tensor* forward(ggml_context* ctx, ggml_tensor* z);
};

struct TinyAutoEncoder {
  TinyEncoder encoder;
  TinyDecoder decoder;

  ggml_context* ctx  = NULL;
  bool decode_only = false;
  ggml_backend_buffer_t params_buffer = NULL;
  ggml_backend_buffer_t compute_buffer = NULL;  // for compute
  struct ggml_allocr* compute_alloc = NULL;

  int memory_buffer_size = 0;
  ggml_type wtype;
  ggml_backend_t backend = NULL;

  TinyAutoEncoder(bool decoder_only_ = true);

  size_t calculate_mem_size();

  bool init(ggml_backend_t backend_);

  void alloc_params();

  void map_by_name(std::map<std::string, ggml_tensor*>& tensors);

  bool load_from_file(const std::string& file_path, ggml_backend_t backend);

  struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph);

  void begin(struct ggml_tensor* x, bool decode);

  void compute(struct ggml_tensor* work_result, const int n_threads, struct ggml_tensor* z, bool decode_graph);

  void end();
};

/*
    ===================================    ESRGAN  ===================================
    References:
    https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/basicsr/archs/rrdbnet_arch.py

*/

struct ResidualDenseBlock {
  int num_features;
  int num_grow_ch;
  ggml_tensor* conv1_w;  // [num_grow_ch, num_features, 3, 3]
  ggml_tensor* conv1_b;  // [num_grow_ch]

  ggml_tensor* conv2_w;  // [num_grow_ch, num_features + num_grow_ch, 3, 3]
  ggml_tensor* conv2_b;  // [num_grow_ch]

  ggml_tensor* conv3_w;  // [num_grow_ch, num_features + 2 * num_grow_ch, 3, 3]
  ggml_tensor* conv3_b;  // [num_grow_ch]

  ggml_tensor* conv4_w;  // [num_grow_ch, num_features + 3 * num_grow_ch, 3, 3]
  ggml_tensor* conv4_b;  // [num_grow_ch]

  ggml_tensor* conv5_w;  // [num_features, num_features + 4 * num_grow_ch, 3, 3]
  ggml_tensor* conv5_b;  // [num_features]

  ResidualDenseBlock();

  ResidualDenseBlock(int num_feat, int n_grow_ch);

  size_t calculate_mem_size();

  int get_num_tensors();

  void init_params(ggml_context* ctx);

  void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix);

  ggml_tensor* forward(ggml_context* ctx, ggml_tensor* out_scale, ggml_tensor* x /* feat */);
};

struct EsrganBlock {
  ResidualDenseBlock rd_blocks[3];
  int num_residual_blocks = 3;

  EsrganBlock();

  EsrganBlock(int num_feat, int num_grow_ch);

  int get_num_tensors();

  size_t calculate_mem_size();

  void init_params(ggml_context* ctx);

  void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix);

  ggml_tensor* forward(ggml_context* ctx, ggml_tensor* out_scale, ggml_tensor* x);
};

struct ESRGAN {
  int scale        = 4;  // default RealESRGAN_x4plus_anime_6B
  int num_blocks   = 6;  // default RealESRGAN_x4plus_anime_6B
  int in_channels  = 3;
  int out_channels = 3;
  int num_features = 64;   // default RealESRGAN_x4plus_anime_6B
  int num_grow_ch  = 32;   // default RealESRGAN_x4plus_anime_6B
  int tile_size    = 128;  // avoid cuda OOM for 4gb VRAM

  ggml_tensor* conv_first_w;  // [num_features, in_channels, 3, 3]
  ggml_tensor* conv_first_b;  // [num_features]

  EsrganBlock body_blocks[6];
  ggml_tensor* conv_body_w;  // [num_features, num_features, 3, 3]
  ggml_tensor* conv_body_b;  // [num_features]

  // upsample
  ggml_tensor* conv_up1_w;  // [num_features, num_features, 3, 3]
  ggml_tensor* conv_up1_b;  // [num_features]
  ggml_tensor* conv_up2_w;  // [num_features, num_features, 3, 3]
  ggml_tensor* conv_up2_b;  // [num_features]

  ggml_tensor* conv_hr_w;    // [num_features, num_features, 3, 3]
  ggml_tensor* conv_hr_b;    // [num_features]
  ggml_tensor* conv_last_w;  // [out_channels, num_features, 3, 3]
  ggml_tensor* conv_last_b;  // [out_channels]

  ggml_context* ctx = NULL;
  bool decode_only = false;
  ggml_backend_buffer_t params_buffer = NULL;
  ggml_backend_buffer_t compute_buffer = NULL;  // for compute
  struct ggml_allocr* compute_alloc = NULL;

  int memory_buffer_size = 0;
  ggml_type wtype;
  ggml_backend_t backend = NULL;

  ESRGAN();

  size_t calculate_mem_size();

  int get_num_tensors();

  bool init(ggml_backend_t backend_);

  void alloc_params();

  bool load_from_file(const std::string& file_path, ggml_backend_t backend);

  void map_by_name(std::map<std::string, ggml_tensor*>& tensors);

  ggml_tensor* forward(ggml_context* ctx0, ggml_tensor* out_scale, ggml_tensor* x /* feat */);

  struct ggml_cgraph* build_graph(struct ggml_tensor* x);

  void begin(struct ggml_tensor* x);

  void compute(struct ggml_tensor* work_result, const int n_threads, struct ggml_tensor* x);

  void end();
};

float ggml_backend_tensor_get_f32(ggml_tensor* tensor);

struct LoraModel {
  float multiplier = 1.0f;
  std::map<std::string, struct ggml_tensor*> lora_tensors;

  struct ggml_context* ctx = NULL;
  ggml_backend_buffer_t params_buffer_lora = NULL;
  ggml_backend_t backend = NULL;

  bool load(ggml_backend_t backend_, std::string file_path);

  struct ggml_cgraph* build_graph(struct ggml_allocr* compute_alloc, std::map<std::string, struct ggml_tensor*> model_tensors);

  void apply(std::map<std::string, struct ggml_tensor*> model_tensors, int n_threads);

  void release();
};

/*================================================= CompVisDenoiser ==================================================*/

// Ref: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py

struct SigmaSchedule {
  float alphas_cumprod[TIMESTEPS];
  float sigmas[TIMESTEPS];
  float log_sigmas[TIMESTEPS];

  virtual std::vector<float> get_sigmas(uint32_t n) = 0;

  float sigma_to_t(float sigma);

  float t_to_sigma(float t);
};

struct DiscreteSchedule : SigmaSchedule {
  std::vector<float> get_sigmas(uint32_t n);
};

struct KarrasSchedule : SigmaSchedule {
  std::vector<float> get_sigmas(uint32_t n);
};

struct Denoiser {
  std::shared_ptr<SigmaSchedule> schedule              = std::make_shared<DiscreteSchedule>();
  virtual std::vector<float> get_scalings(float sigma) = 0;
};

struct CompVisDenoiser : public Denoiser {
  float sigma_data = 1.0f;

  std::vector<float> get_scalings(float sigma);
};

struct CompVisVDenoiser : public Denoiser {
  float sigma_data = 1.0f;

  std::vector<float> get_scalings(float sigma);
};

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
 public:
  SDVersion version;
  bool vae_decode_only         = false;
  bool free_params_immediately = false;

  std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();
  int n_threads            = -1;
  float scale_factor       = 0.18215f;

  FrozenCLIPEmbedderWithCustomWords cond_stage_model;
  UNetModel diffusion_model;
  AutoEncoderKL first_stage_model;
  bool use_tiny_autoencoder = false;
  bool vae_tiling           = false;

  std::map<std::string, struct ggml_tensor*> tensors;

  std::string lora_model_dir;
  // lora_name => multiplier
  std::unordered_map<std::string, float> curr_lora_state;
  std::map<std::string, LoraModel> loras;

  std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();
  ggml_backend_t backend             = NULL;  // general backend
  ggml_type model_data_type          = GGML_TYPE_COUNT;

  TinyAutoEncoder tae_first_stage;
  std::string taesd_path;

  ESRGAN esrgan_upscaler;
  std::string esrgan_path;
  bool upscale_output = false;

  StableDiffusionGGML();

  StableDiffusionGGML(int n_threads,
                      bool vae_decode_only,
                      bool free_params_immediately,
                      std::string lora_model_dir,
                      RNGType rng_type);

  ~StableDiffusionGGML();

  bool load_from_file(const std::string& model_path,
                      const std::string& vae_path,
                      ggml_type wtype,
                      Schedule schedule,
                      int clip_skip);

  bool is_using_v_parameterization_for_sd2(ggml_context* work_ctx);

  void apply_lora(const std::string& lora_name, float multiplier);

  void apply_loras(const std::unordered_map<std::string, float>& lora_state);

  std::pair<ggml_tensor*, ggml_tensor*> get_learned_condition(ggml_context* work_ctx, const std::string& text, int width, int height, bool force_zero_embeddings = false);

  ggml_tensor* sample(ggml_context* work_ctx,
                      ggml_tensor* x_t,
                      ggml_tensor* noise,
                      ggml_tensor* c,
                      ggml_tensor* c_vector,
                      ggml_tensor* uc,
                      ggml_tensor* uc_vector,
                      float cfg_scale,
                      SampleMethod method,
                      const std::vector<float>& sigmas);

  ggml_tensor* get_first_stage_encoding(ggml_context* work_ctx, ggml_tensor* moments);

  ggml_tensor* compute_first_stage(ggml_context* work_ctx, ggml_tensor* x, bool decode);

  uint8_t* upscale(ggml_tensor* image);

  ggml_tensor* encode_first_stage(ggml_context* work_ctx, ggml_tensor* x);

  ggml_tensor* decode_first_stage(ggml_context* work_ctx, ggml_tensor* x);
};

