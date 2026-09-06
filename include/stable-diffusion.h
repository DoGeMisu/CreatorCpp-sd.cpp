#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef SD_BUILD_SHARED_LIB
#define SD_API
#else
#ifdef SD_BUILD_DLL
#define SD_API __declspec(dllexport)
#else
#define SD_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define SD_API __attribute__((visibility("default")))
#else
#define SD_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG,
    CPU_RNG,
    RNG_TYPE_COUNT
};

enum sample_method_t {
    EULER_SAMPLE_METHOD,
    EULER_A_SAMPLE_METHOD,
    HEUN_SAMPLE_METHOD,
    DPM2_SAMPLE_METHOD,
    DPMPP2S_A_SAMPLE_METHOD,
    DPMPP2M_SAMPLE_METHOD,
    DPMPP2Mv2_SAMPLE_METHOD,
    IPNDM_SAMPLE_METHOD,
    IPNDM_V_SAMPLE_METHOD,
    LCM_SAMPLE_METHOD,
    DDIM_TRAILING_SAMPLE_METHOD,
    TCD_SAMPLE_METHOD,
    RES_MULTISTEP_SAMPLE_METHOD,
    RES_2S_SAMPLE_METHOD,
    ER_SDE_SAMPLE_METHOD,
    EULER_CFG_PP_SAMPLE_METHOD,
    EULER_A_CFG_PP_SAMPLE_METHOD,
    EULER_GE_SAMPLE_METHOD,
    DPMPP2M_SDE_SAMPLE_METHOD,
    DPMPP2M_SDE_BT_SAMPLE_METHOD,
    LMS_SAMPLE_METHOD,
    SAMPLE_METHOD_COUNT
};

enum scheduler_t {
    DISCRETE_SCHEDULER,
    KARRAS_SCHEDULER,
    EXPONENTIAL_SCHEDULER,
    AYS_SCHEDULER,
    GITS_SCHEDULER,
    SGM_UNIFORM_SCHEDULER,
    SIMPLE_SCHEDULER,
    SMOOTHSTEP_SCHEDULER,
    KL_OPTIMAL_SCHEDULER,
    LCM_SCHEDULER,
    BONG_TANGENT_SCHEDULER,
    LTX2_SCHEDULER,
    LOGIT_NORMAL_SCHEDULER,
    FLUX2_SCHEDULER,
    FLUX_SCHEDULER,
    BETA_SCHEDULER,
    SCHEDULER_COUNT
};

enum prediction_t {
    EPS_PRED,
    V_PRED,
    EDM_V_PRED,
    FLOW_PRED,
    FLUX_FLOW_PRED,
    SEFI_FLOW_PRED,
    MINIT2I_FLOW_PRED,
    PREDICTION_COUNT
};

// same as enum ggml_type
enum sd_type_t {
    SD_TYPE_F32  = 0,
    SD_TYPE_F16  = 1,
    SD_TYPE_Q4_0 = 2,
    SD_TYPE_Q4_1 = 3,
    // SD_TYPE_Q4_2 = 4, support has been removed
    // SD_TYPE_Q4_3 = 5, support has been removed
    SD_TYPE_Q5_0    = 6,
    SD_TYPE_Q5_1    = 7,
    SD_TYPE_Q8_0    = 8,
    SD_TYPE_Q8_1    = 9,
    SD_TYPE_Q2_K    = 10,
    SD_TYPE_Q3_K    = 11,
    SD_TYPE_Q4_K    = 12,
    SD_TYPE_Q5_K    = 13,
    SD_TYPE_Q6_K    = 14,
    SD_TYPE_Q8_K    = 15,
    SD_TYPE_IQ2_XXS = 16,
    SD_TYPE_IQ2_XS  = 17,
    SD_TYPE_IQ3_XXS = 18,
    SD_TYPE_IQ1_S   = 19,
    SD_TYPE_IQ4_NL  = 20,
    SD_TYPE_IQ3_S   = 21,
    SD_TYPE_IQ2_S   = 22,
    SD_TYPE_IQ4_XS  = 23,
    SD_TYPE_I8      = 24,
    SD_TYPE_I16     = 25,
    SD_TYPE_I32     = 26,
    SD_TYPE_I64     = 27,
    SD_TYPE_F64     = 28,
    SD_TYPE_IQ1_M   = 29,
    SD_TYPE_BF16    = 30,
    // SD_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // SD_TYPE_Q4_0_4_8 = 32,
    // SD_TYPE_Q4_0_8_8 = 33,
    SD_TYPE_TQ1_0 = 34,
    SD_TYPE_TQ2_0 = 35,
    // SD_TYPE_IQ4_NL_4_4 = 36,
    // SD_TYPE_IQ4_NL_4_8 = 37,
    // SD_TYPE_IQ4_NL_8_8 = 38,
    SD_TYPE_MXFP4 = 39,  // MXFP4 (1 block)
    SD_TYPE_NVFP4 = 40,  // NVFP4 (4 blocks, E4M3 scale)
    SD_TYPE_Q1_0  = 41,
    SD_TYPE_COUNT = 42,
};

enum sd_log_level_t {
    SD_LOG_DEBUG,
    SD_LOG_INFO,
    SD_LOG_WARN,
    SD_LOG_ERROR
};

enum preview_t {
    PREVIEW_NONE,
    PREVIEW_PROJ,
    PREVIEW_TAE,
    PREVIEW_VAE,
    PREVIEW_COUNT
};

enum lora_apply_mode_t {
    LORA_APPLY_AUTO,
    LORA_APPLY_IMMEDIATELY,
    LORA_APPLY_AT_RUNTIME,
    LORA_APPLY_MODE_COUNT,
};

typedef struct {
    bool enabled;
    bool temporal_tiling;
    int tile_size_x;
    int tile_size_y;
    float target_overlap;
    float rel_size_x;
    float rel_size_y;
    const char* extra_tiling_args;
} sd_tiling_params_t;

typedef struct {
    const char* name;
    const char* path;
} sd_embedding_t;

enum sd_vae_format_t {
    SD_VAE_FORMAT_AUTO = -1,
    SD_VAE_FORMAT_FLUX,
    SD_VAE_FORMAT_SD3,
    SD_VAE_FORMAT_FLUX2,
    SD_VAE_FORMAT_WAN,
    SD_VAE_FORMAT_COUNT,
};

// ============================================================================
// ABI-compatible struct layout for StableDiffusion.NET 7.0.0 (42 fields).
// DO NOT reorder — the field order must exactly match what SD.NET P/Invoke
// marshals.  See the comparison table in docs/abi-compat.md for details.
// Field names that exist in upstream sd.cpp master but NOT in SD.NET 7.0.0
// are omitted here; the C++ source uses hardcoded defaults for those.
// ============================================================================
typedef struct {
    const char* model_path;                       // 1
    const char* clip_l_path;                      // 2
    const char* clip_g_path;                      // 3
    const char* clip_vision_path;                 // 4
    const char* t5xxl_path;                       // 5
    const char* llm_path;                         // 6
    const char* llm_vision_path;                  // 7
    const char* diffusion_model_path;             // 8
    const char* high_noise_diffusion_model_path;  // 9
    const char* vae_path;                         // 10
    const char* taesd_path;                       // 11
    const char* control_net_path;                 // 12
    const sd_embedding_t* embeddings;             // 13
    uint32_t embedding_count;                     // 14
    const char* photo_maker_path;                 // 15
    const char* tensor_type_rules;                // 16
    bool vae_decode_only;                         // 17  (SD.NET only; unused by C++ core)
    bool free_params_immediately;                 // 18  (SD.NET only; unused by C++ core)
    int n_threads;                                // 19
    enum sd_type_t wtype;                         // 20
    enum rng_type_t rng_type;                     // 21
    enum rng_type_t sampler_rng_type;             // 22
    enum prediction_t prediction;                 // 23
    enum lora_apply_mode_t lora_apply_mode;       // 24
    bool offload_params_to_cpu;                   // 25  (SD.NET only; unused by C++ core)
    bool enable_mmap;                             // 26
    bool keep_clip_on_cpu;                        // 27  (SD.NET only; unused by C++ core)
    bool keep_control_net_on_cpu;                 // 28  (SD.NET only; unused by C++ core)
    bool keep_vae_on_cpu;                         // 29  (SD.NET only; unused by C++ core)
    bool flash_attn;                              // 30
    bool diffusion_flash_attn;                    // 31
    bool tae_preview_only;                        // 32
    bool diffusion_conv_direct;                   // 33
    bool vae_conv_direct;                         // 34
    bool circular_x;                              // 35  (SD.NET only; C++ core uses gen params)
    bool circular_y;                              // 36  (SD.NET only; C++ core uses gen params)
    bool force_sdxl_vae_conv_scale;               // 37
    bool chroma_use_dit_mask;                     // 38  (SD.NET only; unused by C++ core)
    bool chroma_use_t5_mask;                      // 39  (SD.NET only; unused by C++ core)
    int chroma_t5_mask_pad;                       // 40  (SD.NET only; unused by C++ core)
    bool qwen_image_zero_cond_t;                  // 41  (SD.NET only; unused by C++ core)
    float max_vram;                               // 42  (GiB budget, 0=disabled, -1=auto)
} sd_ctx_params_t;

typedef struct {
    uint32_t sample_rate;
    uint32_t channels;
    uint64_t sample_count;
    float* data;
} sd_audio_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;

typedef struct {
    sd_image_t* frames;
    int frame_count;
    int fps;
    sd_audio_t audio;
} sd_ref_video_t;

typedef struct {
    int* layers;
    size_t layer_count;
    float layer_start;
    float layer_end;
    float scale;
} sd_slg_params_t;

typedef struct {
    float txt_cfg;
    float img_cfg;
    float distilled_guidance;
    sd_slg_params_t slg;
} sd_guidance_params_t;

typedef struct {
    sd_guidance_params_t guidance;
    enum scheduler_t scheduler;
    enum sample_method_t sample_method;
    int sample_steps;
    float eta;
    int shifted_timestep;
    float* custom_sigmas;
    int custom_sigmas_count;
    float flow_shift;
    const char* extra_sample_args;
} sd_sample_params_t;

typedef struct {
    sd_image_t* id_images;
    int id_images_count;
    const char* id_embed_path;
    float style_strength;
} sd_pm_params_t;  // photo maker

typedef struct {
    const char* id_embedding_path;
    float id_weight;
} sd_pulid_params_t;

enum sd_cache_mode_t {
    SD_CACHE_DISABLED = 0,
    SD_CACHE_EASYCACHE,
    SD_CACHE_UCACHE,
    SD_CACHE_DBCACHE,
    SD_CACHE_TAYLORSEER,
    SD_CACHE_CACHE_DIT,
    SD_CACHE_SPECTRUM,
};

typedef struct {
    enum sd_cache_mode_t mode;
    float reuse_threshold;
    float start_percent;
    float end_percent;
    float error_decay_rate;
    bool use_relative_threshold;
    bool reset_error_on_compute;
    int Fn_compute_blocks;
    int Bn_compute_blocks;
    float residual_diff_threshold;
    int max_warmup_steps;
    int max_cached_steps;
    int max_continuous_cached_steps;
    int taylorseer_n_derivatives;
    int taylorseer_skip_interval;
    const char* scm_mask;
    bool scm_policy_dynamic;
    float spectrum_w;
    int spectrum_m;
    float spectrum_lam;
    int spectrum_window_size;
    float spectrum_flex_window;
    int spectrum_warmup_steps;
    float spectrum_stop_percent;
} sd_cache_params_t;

typedef struct {
    bool is_high_noise;
    float multiplier;
    const char* path;
} sd_lora_t;

enum sd_hires_upscaler_t {
    SD_HIRES_UPSCALER_NONE,
    SD_HIRES_UPSCALER_LATENT,
    SD_HIRES_UPSCALER_LATENT_NEAREST,
    SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT,
    SD_HIRES_UPSCALER_LATENT_ANTIALIASED,
    SD_HIRES_UPSCALER_LATENT_BICUBIC,
    SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED,
    SD_HIRES_UPSCALER_LANCZOS,
    SD_HIRES_UPSCALER_NEAREST,
    SD_HIRES_UPSCALER_MODEL,
    SD_HIRES_UPSCALER_COUNT,
};

typedef struct {
    bool enabled;
    enum sd_hires_upscaler_t upscaler;
    const char* model_path;
    float scale;
    int target_width;
    int target_height;
    int steps;
    float denoising_strength;
    int upscale_tile_size;
    float* custom_sigmas;
    int custom_sigmas_count;
} sd_hires_params_t;

typedef struct {
    const sd_lora_t* loras;
    uint32_t lora_count;
    const char* prompt;
    const char* negative_prompt;
    int clip_skip;
    sd_image_t init_image;
    sd_image_t* ref_images;
    int ref_images_count;
    const char* ref_image_args;
    sd_image_t mask_image;
    int width;
    int height;
    sd_sample_params_t sample_params;
    float strength;
    int64_t seed;
    int batch_count;
    sd_image_t control_image;
    float control_strength;
    sd_image_t ip_adapter_image;
    float ip_adapter_strength;
    sd_pm_params_t pm_params;
    sd_pulid_params_t pulid_params;
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
    sd_hires_params_t hires;
    int qwen_image_layers;
    bool circular_x;
    bool circular_y;
} sd_img_gen_params_t;

typedef struct {
    const sd_lora_t* loras;
    uint32_t lora_count;
    const char* prompt;
    const char* negative_prompt;
    int clip_skip;
    sd_image_t init_image;
    sd_image_t end_image;
    sd_image_t* ref_images;
    int ref_images_count;
    sd_ref_video_t* ref_videos;
    int ref_videos_count;
    sd_audio_t* ref_audios;
    int ref_audios_count;
    sd_image_t* control_frames;
    int control_frames_size;
    int width;
    int height;
    sd_sample_params_t sample_params;
    sd_sample_params_t high_noise_sample_params;
    float moe_boundary;
    float strength;
    int64_t seed;
    int video_frames;
    int fps;
    float vace_strength;
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
    sd_hires_params_t hires;
    bool circular_x;
    bool circular_y;
} sd_vid_gen_params_t;

typedef struct sd_ctx_t sd_ctx_t;
struct ggml_tensor;

typedef void (*sd_log_cb_t)(enum sd_log_level_t level, const char* text, void* data);
typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);
typedef void (*sd_preview_cb_t)(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data);
typedef bool (*sd_graph_eval_callback_t)(struct ggml_tensor* t, bool ask, void* user_data);

SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
SD_API void sd_set_preview_callback(sd_preview_cb_t cb, enum preview_t mode, int interval, bool denoised, bool noisy, void* data);
SD_API void sd_set_backend_eval_callback(sd_graph_eval_callback_t cb, void* data);
SD_API int32_t sd_get_num_physical_cores();
SD_API const char* sd_get_system_info();
SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx);
SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx);

// ControlNet hot-swap APIs are not safe to call while generation is in flight.
SD_API bool sd_ctx_load_control_net(sd_ctx_t* sd_ctx, const char* path);
SD_API bool sd_ctx_unload_control_net(sd_ctx_t* sd_ctx);
SD_API bool sd_ctx_has_control_net(const sd_ctx_t* sd_ctx);

// Set ControlNet weight type override for subsequent sd_ctx_load_control_net calls.
// Pass SD_TYPE_COUNT to disable override (use original tensor types).
SD_API void sd_ctx_set_control_net_wtype(sd_ctx_t* sd_ctx, enum sd_type_t wtype);

SD_API const char* sd_type_name(enum sd_type_t type);
SD_API enum sd_type_t str_to_sd_type(const char* str);
SD_API const char* sd_rng_type_name(enum rng_type_t rng_type);
SD_API enum rng_type_t str_to_rng_type(const char* str);
SD_API const char* sd_sample_method_name(enum sample_method_t sample_method);
SD_API enum sample_method_t str_to_sample_method(const char* str);
SD_API const char* sd_scheduler_name(enum scheduler_t scheduler);
SD_API enum scheduler_t str_to_scheduler(const char* str);
SD_API const char* sd_prediction_name(enum prediction_t prediction);
SD_API enum prediction_t str_to_prediction(const char* str);
SD_API const char* sd_preview_name(enum preview_t preview);
SD_API enum preview_t str_to_preview(const char* str);
SD_API const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode);
SD_API enum lora_apply_mode_t str_to_lora_apply_mode(const char* str);
SD_API const char* sd_hires_upscaler_name(enum sd_hires_upscaler_t upscaler);
SD_API enum sd_hires_upscaler_t str_to_sd_hires_upscaler(const char* str);

SD_API void sd_cache_params_init(sd_cache_params_t* cache_params);
SD_API void sd_hires_params_init(sd_hires_params_t* hires_params);

SD_API void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params);
SD_API char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params);

SD_API sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params);
SD_API void free_sd_ctx(sd_ctx_t* sd_ctx);
SD_API void free_sd_audio(sd_audio_t* audio);

SD_API void sd_sample_params_init(sd_sample_params_t* sample_params);
SD_API char* sd_sample_params_to_str(const sd_sample_params_t* sample_params);

SD_API enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx);
SD_API enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method);

SD_API void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params);
SD_API char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params);
SD_API bool generate_image(sd_ctx_t* sd_ctx,
                           const sd_img_gen_params_t* sd_img_gen_params,
                           sd_image_t** images_out,
                           int* num_images_out);

enum sd_cancel_mode_t {
    // Stop the current generation as soon as possible.
    SD_CANCEL_ALL,
    // Finish the current image sample, then skip additional batch latents and return completed images.
    SD_CANCEL_NEW_LATENTS,
    // Clear a pending cancellation request.
    SD_CANCEL_RESET
};

SD_API void sd_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode);

SD_API void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params);
SD_API bool generate_video(sd_ctx_t* sd_ctx,
                           const sd_vid_gen_params_t* sd_vid_gen_params,
                           sd_image_t** frames_out,
                           int* num_frames_out,
                           sd_audio_t** audio_out);

// ===========================================================================
// Two-phase video generation support (memory-light mode)
//
// Phase 1 (sd_encode_video_prompt): load ONLY the text encoder for the chosen
// video model architecture (e.g. Gemma-3-12B LLM + embeddings connectors for
// LTX-2.3, T5 XXL for Wan2.x, MLLM for HunyuanVideo), encode the prompt into
// a conditioning embedding, serialize it to disk, then release ALL text-encoder
// memory. Returns true on success.
//
// Phase 2 (sd_ctx_set_precomputed_embeddings + generate_video): load ONLY the
// diffusion model + VAE (no text encoder). generate_video() then reads the
// precomputed conditioning embeddings from disk and skips the text-encoder
// pass entirely.
//
// This keeps the large text encoder and the diffusion model from being
// resident at the same time, which is essential on 8GB-VRAM machines.
// ===========================================================================

// Text encoder architectures supported by the two-phase pipeline.
enum sd_video_encoder_type_t {
    SD_VIDEO_ENCODER_LTX2 = 0,  // LTX-2.3: Gemma-3-12B LLM + embeddings connectors
    SD_VIDEO_ENCODER_WAN,       // Wan2.x: T5 XXL
    SD_VIDEO_ENCODER_HUNYUAN,   // HunyuanVideo: MLLM (text encoder)
    SD_VIDEO_ENCODER_MINIMAX,   // MiniMax-H3: LLM text encoder
    SD_VIDEO_ENCODER_COUNT
};

// Phase 1 (generic): Encode a prompt to a conditioning embedding file.
//   encoder_type:  which text-encoder architecture to use (see enum above).
//   te_path:       text encoder model path (Gemma LLM for LTX2, T5 XXL for Wan,
//                  MLLM for HunyuanVideo/MiniMax-H3).
//   te_extra_path: extra weights required by the encoder (LTX-2.3 embeddings
//                  connectors; may be NULL for models that do not need it).
//   prompt / negative_prompt: the conditioning texts (negative_prompt may be NULL).
//   n_threads:     number of CPU threads for the encoder.
//   cond_out_path / uncond_out_path: output files (raw float32 blobs, "LTXE" header).
// The embedding files are raw float32 blobs with an "LTXE" header.
SD_API bool sd_encode_video_prompt(enum sd_video_encoder_type_t encoder_type,
                                   const char* te_path,
                                   const char* te_extra_path,
                                   const char* prompt,
                                   const char* negative_prompt,
                                   int n_threads,
                                   const char* cond_out_path,
                                   const char* uncond_out_path);

// Phase 1 (LTX-2.3 convenience wrapper; equivalent to
// sd_encode_video_prompt(SD_VIDEO_ENCODER_LTX2, ...)).
SD_API bool sd_encode_ltxav_prompt(const char* llm_path,
                                   const char* embeddings_connectors_path,
                                   const char* prompt,
                                   const char* negative_prompt,
                                   int n_threads,
                                   const char* cond_out_path,
                                   const char* uncond_out_path);

// Phase 2: Tell a generation context to use precomputed embeddings produced by
// sd_encode_video_prompt()/sd_encode_ltxav_prompt(). After this call,
// generate_video() skips the text encoder and reads the conditioning directly
// from disk. The context must be created WITHOUT a text encoder path
// (diffusion model + VAE only). Returns true on success.
SD_API bool sd_ctx_set_precomputed_embeddings(sd_ctx_t* sd_ctx,
                                              const char* cond_embedding_path,
                                              const char* uncond_embedding_path);

// ===========================================================================
// Three-phase video generation support (maximum VRAM efficiency)
//
// Phase 1 (sd_encode_video_prompt): Text encoder only — encode, save, release.
// Phase 2 (generate_video with decode_only skip): Diffusion only — sample,
//        save latent, release. VAE weights are NOT loaded, giving Diffusion
//        exclusive access to all available VRAM.
// Phase 3 (sd_decode_video_latent): VAE only — load VAE, decode latent, release.
//
// This prevents the VAE (~1.7 GB for LTX-2.3) from occupying VRAM during the
// diffusion sampling loop, which on 8 GB GPUs reduces graph-cut segments and
// eliminates redundant RAM→VRAM weight staging.
// ===========================================================================

// Phase 2 flag: Tell a diffusion context to skip VAE decode in generate_video().
// After generate_video() returns, use sd_save_video_latent() to persist the
// raw latent tensor to disk, then free the diffusion context.
// The context must be created with new_video_diffusion_ctx().
// Returns true on success.
SD_API bool sd_ctx_set_decode_only(sd_ctx_t* sd_ctx, bool decode_only);

// Phase 2: Create a diffusion-only context (no VAE, no text encoder).
//   diffusion_model_path: diffusion model file (e.g. ltx-2.3-22b.gguf).
//   n_threads:           CPU thread count.
//   max_vram:           max VRAM in GiB (0 = auto-detect).
//   flash_attn:         enable flash attention in diffusion model.
// Returns NULL on failure.
// Use sd_ctx_set_precomputed_embeddings() to inject TE embeddings,
// then call generate_video() with sd_ctx_set_decode_only(true) to
// sample and cache the latent without VAE decode.
SD_API sd_ctx_t* new_video_diffusion_ctx(const char* diffusion_model_path,
                                          int n_threads,
                                          int max_vram,
                                          bool flash_attn);

// Phase 2 output: Serialize the final latent produced by generate_video()
// to a .bin file (raw float32, header = "SDLT" + 4×int64 shape).
// Must be called immediately after generate_video() returns true and before
// free_sd_ctx(). Returns true on success.
SD_API bool sd_save_video_latent(sd_ctx_t* sd_ctx, const char* latent_out_path);

// Phase 3: Create a VAE-only context (no diffusion model, no text encoder).
//   vae_path:        video VAE model file.
//   audio_vae_path:  audio VAE model file (may be NULL to skip audio).
//   n_threads:       CPU thread count.
//   max_vram:        max VRAM in GiB (0 = auto-detect).
// Returns NULL on failure.
SD_API sd_ctx_t* new_video_vae_ctx(const char* vae_path,
                                    const char* audio_vae_path,
                                    int n_threads,
                                    int max_vram);

// Phase 3: Decode a latent file (produced by sd_save_video_latent) into video
// frames + optional audio. The latent is loaded from disk, decoded by the VAE
// context, and returned as sd_image_t frames.
//   sd_ctx:          VAE-only context from new_video_vae_ctx().
//   latent_path:     path to .bin file from sd_save_video_latent().
//   width/height:     target video resolution (must match the original request).
//   vae_scale_factor: typically 8 for LTX (use sd_ctx->sd->get_vae_scale_factor()).
//   audio_length:     audio latent length (0 if no audio).
//   fps:              frame rate (used for audio alignment).
//   frames_out:       output frame array (caller frees with free_sd_images).
//   num_frames_out:  output frame count.
//   audio_out:        output audio (may be NULL; caller frees with free_sd_audio).
// Returns true on success.
SD_API bool sd_decode_video_latent(sd_ctx_t* sd_ctx,
                                    const char* latent_path,
                                    int width,
                                    int height,
                                    int vae_scale_factor,
                                    int audio_length,
                                    float fps,
                                    sd_image_t** frames_out,
                                    int* num_frames_out,
                                    sd_audio_t** audio_out);

typedef struct upscaler_ctx_t upscaler_ctx_t;

SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path,
                                        bool direct,
                                        int n_threads,
                                        int tile_size,
                                        const char* backend,
                                        const char* params_backend);
SD_API void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);

SD_API bool upscale(upscaler_ctx_t* upscaler_ctx,
                    sd_image_t input_image,
                    uint32_t upscale_factor,
                    sd_image_t** images_out,
                    int* num_images_out);

SD_API int get_upscale_factor(upscaler_ctx_t* upscaler_ctx);

typedef struct adetailer_ctx_t adetailer_ctx_t;

typedef struct {
    const char* prompt;
    const char* negative_prompt;
    const char* extra_ad_args;
} sd_adetailer_params_t;

SD_API adetailer_ctx_t* new_adetailer_ctx(const char* detector_path,
                                          int n_threads,
                                          const char* backend,
                                          const char* params_backend);
SD_API void free_adetailer_ctx(adetailer_ctx_t* adetailer_ctx);
SD_API bool adetail_image(adetailer_ctx_t* adetailer_ctx,
                          sd_ctx_t* sd_ctx,
                          sd_image_t input_image,
                          const sd_adetailer_params_t* adetailer_params,
                          const sd_img_gen_params_t* inpaint_params,
                          sd_image_t** images_out,
                          int* num_images_out);

SD_API bool convert(const char* input_path,
                    const char* vae_path,
                    const char* output_path,
                    enum sd_type_t output_type,
                    const char* tensor_type_rules,
                    bool convert_name);

SD_API bool convert_with_components(const char* model_path,
                                    const char* clip_l_path,
                                    const char* clip_g_path,
                                    const char* t5xxl_path,
                                    const char* diffusion_model_path,
                                    const char* vae_path,
                                    const char* output_path,
                                    enum sd_type_t output_type,
                                    const char* tensor_type_rules,
                                    bool convert_name,
                                    int n_threads);

SD_API bool preprocess_canny(sd_image_t image,
                             float high_threshold,
                             float low_threshold,
                             float weak,
                             float strong,
                             bool inverse);

SD_API bool load_imatrix(const char* imatrix_path);
SD_API void save_imatrix(const char* imatrix_path);
SD_API void enable_imatrix_collection(void);
SD_API void disable_imatrix_collection(void);

SD_API const char* sd_commit(void);
SD_API const char* sd_version(void);

// List available ggml backend devices, one `name<TAB>description` per line.
// The names are the device names accepted by the --backend / --params-backend
// assignment specs. Returns the number of bytes required, excluding the null
// terminator. Passing nullptr or buffer_size 0 only queries the required size.
SD_API size_t sd_list_devices(char* buffer, size_t buffer_size);

// for C API, caller needs to call free_sd_images to free the memory after use
// This helps avoid CRT problems on Windows when memory is allocated in the library but freed in the caller, which may use a different CRT.
SD_API void free_sd_images(sd_image_t* result_images, int num_images);

// ===========================================================================
// LoRA Training Support API
// These functions expose internal model components for external LoRA training
// libraries (e.g. sd_train.dll). They allow:
//   1. Extracting UNet weight tensors as F32 buffers
//   2. Encoding images to VAE latents
//   3. Encoding text prompts to CLIP/T5 conditioning embeddings
// ===========================================================================

// --- UNet weight extraction ---
// Get the number of named parameter tensors in the UNet (diffusion model).
// Returns 0 if the context is invalid or the model is not loaded.
SD_API int sd_get_unet_param_count(const sd_ctx_t* sd_ctx);

// Get the name of the i-th UNet parameter tensor.
// Returns NULL on invalid index. The returned pointer is owned by the context
// and remains valid until the next call. Pass buffer/buffer_size to receive
// a copy; if buffer is NULL or buffer_size is 0, only the required length is
// returned (excluding null terminator).
SD_API size_t sd_get_unet_param_name(const sd_ctx_t* sd_ctx, int index,
                                      char* buffer, size_t buffer_size);

// Get the shape of the i-th UNet parameter tensor.
// Writes up to 4 dims into dims_out and returns the number of dims, or 0 on error.
SD_API int sd_get_unet_param_shape(const sd_ctx_t* sd_ctx, int index,
                                    int64_t dims_out[4]);

// Copy the i-th UNet parameter tensor data as F32 into the caller's buffer.
// The buffer must hold at least sd_get_unet_param_count_floats(...) floats.
// Returns the number of floats copied, or 0 on error.
// Note: quantized tensors are dequantized to F32 on the fly.
SD_API int64_t sd_get_unet_param_data(const sd_ctx_t* sd_ctx, int index,
                                       float* out_buffer, int64_t buffer_size);

// Convenience: get the number of floats in the i-th tensor (= product of dims).
SD_API int64_t sd_get_unet_param_numel(const sd_ctx_t* sd_ctx, int index);

// --- VAE encode ---
// Encode an image to VAE latent space.
// image_data: RGB pixel data, [H * W * 3] floats in [0, 1] range.
// width, height: image dimensions (must be multiples of 8).
// out_latent: caller-provided buffer for the latent output.
//   Latent shape is [C=4, H/8, W/8] for SD 1.x, so buffer_size = 4 * (H/8) * (W/8).
// Returns the number of floats written, or 0 on error.
SD_API int64_t sd_vae_encode(const sd_ctx_t* sd_ctx,
                              const float* image_data,
                              int width, int height,
                              float* out_latent, int64_t buffer_size);

// --- Text encode (CLIP/T5) ---
// Encode a text prompt to conditioning embeddings.
// text: null-terminated prompt string.
// clip_skip: number of CLIP layers to skip (-1 = no skip).
// out_crossattn: buffer for c_crossattn embeddings (context for cross-attention).
// out_crossattn_size: [in] buffer capacity in floats, [out] actual floats written.
// out_vector: buffer for c_vector embeddings (pooled output, may be NULL).
// out_vector_size: [in] buffer capacity, [out] actual floats written.
// Returns true on success.
SD_API bool sd_text_encode(const sd_ctx_t* sd_ctx,
                            const char* text,
                            int clip_skip,
                            float* out_crossattn, int64_t* out_crossattn_size,
                            float* out_vector, int64_t* out_vector_size);

#ifdef __cplusplus
}
#endif

#endif  // __STABLE_DIFFUSION_H__
