//
// Created by Stefan on 1/1/2024.
//

#include "dl/diffusion_model.hpp"
#include "dl/sd_ggml.hpp"

#include "util/util.hpp"

#include <stable-diffusion.h>

namespace dl {

DiffusionModel::DiffusionModel(const DiffusionModelProps& props) {
  m_StableDiffusionGGMLBackend = std::make_shared<StableDiffusionGGML>(
      -1, false, false, "", RNGType::STD_DEFAULT_RNG);

  m_StableDiffusionGGMLBackend->use_tiny_autoencoder = false;
  m_StableDiffusionGGMLBackend->taesd_path = "";
  m_StableDiffusionGGMLBackend->upscale_output = false;
  m_StableDiffusionGGMLBackend->esrgan_path = "";
  m_StableDiffusionGGMLBackend->vae_tiling = false;

  m_StableDiffusionGGMLBackend->load_from_file(
      props.Checkpoint.string(), props.VariationalAutoencoder.string(),
      GGML_TYPE_F16, Schedule::DEFAULT, -1);
}

std::experimental::generator<ggml_tensor*> DiffusionModel::Generate(
    std::string prompt) {
  std::string negative_prompt;
  float cfg_scale = 7.0f;
  int width = 512, height = 512;
  SampleMethod sample_method = SampleMethod::EULER_A;
  int sample_steps = 20;
  int64_t seed = 69420;
  int batch_count = 1;

  std::vector<uint8_t*> results;

  auto result_pair = extract_and_remove_lora(prompt);
  std::unordered_map<std::string, float> lora_f2m = result_pair.first;
  prompt = result_pair.second;

  m_StableDiffusionGGMLBackend->apply_loras(lora_f2m);

  ggml_init_params params = {
      .mem_size = (static_cast<size_t>(10 * 1024 * 1024) +
                   width * height * 3 * sizeof(float)) *
                  batch_count,
      .mem_buffer = nullptr,
      .no_alloc = false};

  ggml_context* work_ctx = ggml_init(params);
  util::error::ErrNDie(!work_ctx, "Failed to create a GGML work context.");

  // ggml_context* work_sec_ctx = ggml_init(params);
  // util::error::ErrNDie(!work_ctx, "Failed to create a GGML work context.");

  auto [c, c_vector] = m_StableDiffusionGGMLBackend->get_learned_condition(
      work_ctx, prompt, width, height);
  ggml_tensor* uc = nullptr;
  ggml_tensor* uc_vector = nullptr;

  if (cfg_scale != 1.0) {
    bool force_zero_embeddings = false;
    if (m_StableDiffusionGGMLBackend->version == VERSION_XL &&
        negative_prompt.size() == 0) {
      force_zero_embeddings = true;
    }
    auto uncond_pair = m_StableDiffusionGGMLBackend->get_learned_condition(
        work_ctx, negative_prompt, width, height, force_zero_embeddings);
    uc = uncond_pair.first;
    uc_vector = uncond_pair.second;  // [adm_in_channels, ]
  }

  if (m_StableDiffusionGGMLBackend->free_params_immediately) {
    m_StableDiffusionGGMLBackend->cond_stage_model.destroy();
  }

  // preparations are done

  std::vector<ggml_tensor*> final_latents;
  int C = 4;
  int W = width / 8;
  int H = height / 8;

  // TODO
  for (int batch : std::views::iota(0, batch_count)) {
    int64_t sampling_start = ggml_time_ms();
    int64_t cur_seed = seed + batch;
    std::print("generating image: {}/{} - seed {}\n", batch + 1, batch_count,
               cur_seed);

    m_StableDiffusionGGMLBackend->rng->manual_seed(cur_seed);
    ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
    ggml_tensor_set_f32_randn(x_t, m_StableDiffusionGGMLBackend->rng);

    std::vector<float> sigmas =
        m_StableDiffusionGGMLBackend->denoiser->schedule->get_sigmas(
            sample_steps);

    // let's try and copy the sample data to this
    m_SampleCopy = nullptr;

    // the sampling loop todo
    for (ggml_tensor* sample :
         Sample(work_ctx, x_t, nullptr, c, c_vector, uc, uc_vector, cfg_scale,
                sample_method, sigmas) |
             std::views::take(sample_steps + 2)) {
      if (m_SampleCopy == nullptr) {
        m_SampleCopy = ggml_dup_tensor(work_ctx, sample);
      }
      co_yield sample;
    }

    int64_t sampling_end = ggml_time_ms();
    std::print("sampling completed, taking {}s\n",
               static_cast<float>(sampling_end - sampling_start) * 1.0f / 1000);
    std::cout << std::flush;
  }

  // ggml_free(work_ctx); TODO
}

ggml_tensor* DiffusionModel::Decode(ggml_context* work_ctx,
                                    ggml_tensor* input) {
  ggml_tensor* img =
      m_StableDiffusionGGMLBackend->decode_first_stage(work_ctx, input);
  return img;
}

uint8_t* DiffusionModel::Upscale(ggml_tensor* image) {
  uint8_t* upscaled = nullptr;

  if (m_StableDiffusionGGMLBackend->upscale_output) {
    upscaled = m_StableDiffusionGGMLBackend->upscale(image);
  } else {
    upscaled = sd_tensor_to_image(image);
  }

  auto* final_image = new uint8_t[512 * 512 * 4];  // TODO
  for (int i = 0; i < 512 * 512; ++i) {
    final_image[i * 4] = upscaled[i * 3];
    final_image[i * 4 + 1] = upscaled[i * 3 + 1];
    final_image[i * 4 + 2] = upscaled[i * 3 + 2];
    final_image[i * 4 + 3] = 255;
  }

  return final_image;
}

uint8_t* DiffusionModel::ExtractSample(ggml_tensor* sample) {
  // copy the data from the current sample
  copy_ggml_tensor(m_SampleCopy, sample);

  // apply the decoding and yield
  ggml_context* work_sec_ctx =
      ggml_init({.mem_size = static_cast<size_t>(10 * 1024 * 1024) +
                             512 * 512 * 3 * sizeof(float),
                 .mem_buffer = nullptr,
                 .no_alloc = false});
  util::error::ErrNDie(!work_sec_ctx, "Failed to create a GGML work context.");

  auto* decoded = Decode(work_sec_ctx, m_SampleCopy);
  auto* image = Upscale(decoded);

  ggml_free(work_sec_ctx);

  return image;
}

std::experimental::generator<ggml_tensor*> DiffusionModel::Sample(
    ggml_context* work_ctx, ggml_tensor* x_t, ggml_tensor* initial_noise,
    ggml_tensor* c, ggml_tensor* c_vector, ggml_tensor* uc,
    ggml_tensor* uc_vector, float cfg_scale, SampleMethod method,
    const std::vector<float>& sigmas) {
  int steps = sigmas.size() - 1;

  ggml_tensor* x = ggml_dup_tensor(work_ctx, x_t);
  copy_ggml_tensor(x, x_t);

  ggml_tensor* noised_input = ggml_dup_tensor(work_ctx, x_t);
  ggml_tensor* timesteps =
      ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);  // [N, ]
  ggml_tensor* t_emb =
      new_timestep_embedding(work_ctx, nullptr, timesteps,
                             m_StableDiffusionGGMLBackend->diffusion_model
                                 .model_channels);  // [N, model_channels]
  m_StableDiffusionGGMLBackend->diffusion_model.begin(noised_input, c, t_emb,
                                                      c_vector);

  bool has_unconditioned = cfg_scale != 1.0 && uc != nullptr;

  if (initial_noise == nullptr) {
    // x = x * sigmas[0]
    ggml_tensor_scale(x, sigmas[0]);
  } else {
    // xi = x + noise * sigma_sched[0]
    ggml_tensor_scale(initial_noise, sigmas[0]);
    ggml_tensor_add(x, initial_noise);
  }

  // denoise wrapper
  ggml_tensor* out_cond = ggml_dup_tensor(work_ctx, x);
  ggml_tensor* out_uncond = nullptr;
  if (has_unconditioned) {
    out_uncond = ggml_dup_tensor(work_ctx, x);
  }
  ggml_tensor* denoised = ggml_dup_tensor(work_ctx, x);

  auto denoise = [&](ggml_tensor* input, float sigma, int step) {
    if (step == 1) {
      pretty_progress(0, (int)steps, 0);
    }
    int64_t t0 = ggml_time_us();

    float c_skip = 1.0f;
    float c_out = 1.0f;
    float c_in = 1.0f;
    std::vector<float> scaling =
        m_StableDiffusionGGMLBackend->denoiser->get_scalings(sigma);

    if (scaling.size() == 3) {  // CompVisVDenoiser
      c_skip = scaling[0];
      c_out = scaling[1];
      c_in = scaling[2];
    } else {  // CompVisDenoiser
      c_out = scaling[0];
      c_in = scaling[1];
    }

    float t =
        m_StableDiffusionGGMLBackend->denoiser->schedule->sigma_to_t(sigma);
    ggml_set_f32(timesteps, t);
    set_timestep_embedding(
        timesteps, t_emb,
        m_StableDiffusionGGMLBackend->diffusion_model.model_channels);

    copy_ggml_tensor(noised_input, input);
    // noised_input = noised_input * c_in
    ggml_tensor_scale(noised_input, c_in);

    // cond
    m_StableDiffusionGGMLBackend->diffusion_model.compute(
        out_cond, m_StableDiffusionGGMLBackend->n_threads, noised_input,
        nullptr, c, t_emb, c_vector);

    float* negative_data = nullptr;
    if (has_unconditioned) {
      // uncond
      m_StableDiffusionGGMLBackend->diffusion_model.compute(
          out_uncond, m_StableDiffusionGGMLBackend->n_threads, noised_input,
          nullptr, uc, t_emb, uc_vector);
      negative_data = (float*)out_uncond->data;
    }
    auto* vec_denoised = (float*)denoised->data;
    auto* vec_input = (float*)input->data;
    auto* positive_data = (float*)out_cond->data;
    int ne_elements = (int)ggml_nelements(denoised);
    for (int i = 0; i < ne_elements; i++) {
      float latent_result = positive_data[i];
      if (has_unconditioned) {
        // out_uncond + cfg_scale * (out_cond - out_uncond)
        latent_result = negative_data[i] +
                        cfg_scale * (positive_data[i] - negative_data[i]);
      }
      // v = latent_result, eps = latent_result
      // denoised = (v * c_out + input * c_skip) or (input + eps * c_out)
      vec_denoised[i] = latent_result * c_out + vec_input[i] * c_skip;
    }
    int64_t t1 = ggml_time_us();
    if (step > 0) {
      pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);
      // LOG_INFO("step %d sampling completed taking %.2fs", step, (t1 - t0)
      // * 1.0f / 1000000);
    }
  };

  switch (method) {
    case EULER_A: {
      ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
      ggml_tensor* d = ggml_dup_tensor(work_ctx, x);

      for (int step : std::views::iota(0, steps)) {
        float sigma = sigmas[step];

        // denoise
        denoise(x, sigma, step + 1);

        // d = (x - denoised) / sigma
        {
          auto* vec_d = (float*)d->data;
          auto* vec_x = (float*)x->data;
          auto* vec_denoised = (float*)denoised->data;

          for (int i = 0; i < ggml_nelements(d); i++) {
            vec_d[i] = (vec_x[i] - vec_denoised[i]) / sigma;
          }
        }

        // get_ancestral_step
        float sigma_up = std::min(
            sigmas[step + 1], std::sqrt(sigmas[step + 1] * sigmas[step + 1] *
                                        (sigmas[step] * sigmas[step] -
                                         sigmas[step + 1] * sigmas[step + 1]) /
                                        (sigmas[step] * sigmas[step])));
        float sigma_down = std::sqrt(sigmas[step + 1] * sigmas[step + 1] -
                                     sigma_up * sigma_up);

        // Euler method
        float dt = sigma_down - sigmas[step];
        // x = x + d * dt
        {
          auto* vec_d = (float*)d->data;
          auto* vec_x = (float*)x->data;

          for (int i = 0; i < ggml_nelements(x); i++) {
            vec_x[i] = vec_x[i] + vec_d[i] * dt;
          }
        }

        if (sigmas[step + 1] > 0) {
          // x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise *
          // sigma_up
          ggml_tensor_set_f32_randn(noise, m_StableDiffusionGGMLBackend->rng);
          // noise = load_tensor_from_file(work_ctx, "./rand" +
          // std::to_string(i+1) + ".bin");
          {
            auto* vec_x = (float*)x->data;
            auto* vec_noise = (float*)noise->data;

            for (int i = 0; i < ggml_nelements(x); i++) {
              vec_x[i] = vec_x[i] + vec_noise[i] * sigma_up;
            }
          }
        }

        co_yield x;
      }
    } break;
    case EULER:
      break;
    case HEUN:
      break;
    case DPM2:
      break;
    case DPMPP2S_A:
      break;
    case DPMPP2M:
      break;
    case DPMPP2Mv2:
      break;
    case LCM:
      break;
    case N_SAMPLE_METHODS:
      break;
  }

  //m_StableDiffusionGGMLBackend->diffusion_model.end();
  co_yield x;
}

DiffusionModel::~DiffusionModel() = default;

}  // namespace dl
