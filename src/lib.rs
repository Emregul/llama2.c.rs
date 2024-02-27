#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::time::{SystemTime, UNIX_EPOCH};

mod llamac {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub struct TransformerSystem {
    transformer: Transformer,
    tokenizer: Tokenizer,
    sampler: Sampler,
}

impl TransformerSystem {
    pub fn new(
        checkpoint_path: &str,
        tokenizer_path: &str,
        temperature: f32,
        topp: f32,
        rng_seed: Option<u64>,
    ) -> Self {
        let transformer = Transformer::new(checkpoint_path);
        let vocab_size = transformer.inner.config.vocab_size;
        let tokenizer = Tokenizer::new(tokenizer_path, vocab_size);

        let rng_seed = rng_seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_secs()
        });

        let sampler = Sampler::new(vocab_size, temperature, topp, rng_seed);

        TransformerSystem {
            transformer,
            tokenizer,
            sampler,
        }
    }

    pub fn generate<F: FnMut(&str)>(&mut self, prompt: &str, steps: i32, mut process: F) -> u64 {
        let tokens = self.tokenizer.encode(prompt);
        let mut current = tokens[0];
        let mut next;
        let mut count_tokens = 0;

        for pos in 0..steps {
            let logits = self.transformer.forward(current, pos);

            if pos < (tokens.len() - 1) as i32 {
                next = tokens[(pos + 1) as usize];
            } else {
                next = self.sampler.sample(logits);
            }

            if next == 1 {
                break;
            }

            let piece = self.tokenizer.decode(current, next);
            process(&piece);
            current = next;
            count_tokens += 1;
        }

        count_tokens
    }
}

struct Transformer {
    inner: llamac::Transformer,
}

impl Transformer {
    fn new(checkpoint_path: &str) -> Self {
        let mut transformer: llamac::Transformer = unsafe { std::mem::zeroed() };
        let checkpoint_path = std::ffi::CString::new(checkpoint_path).expect("CString::new failed");

        unsafe {
            llamac::build_transformer(
                &mut transformer as *mut _,
                checkpoint_path.as_ptr() as *mut _,
            );
        }

        Transformer { inner: transformer }
    }

    fn forward(&mut self, token: i32, pos: i32) -> &[f32] {
        let result = unsafe { llamac::forward(&mut self.inner as *mut _, token, pos) };
        unsafe { std::slice::from_raw_parts(result, self.inner.config.vocab_size as usize) }
    }
}

impl Drop for Transformer {
    fn drop(&mut self) {
        unsafe {
            llamac::free_transformer(&mut self.inner as *mut _);
        }
    }
}

struct Tokenizer {
    inner: llamac::Tokenizer,
}

impl Tokenizer {
    fn new(tokenizer_path: &str, vocab_size: i32) -> Self {
        let mut tokenizer: llamac::Tokenizer = unsafe { std::mem::zeroed() };

        let tokenizer_path =
            std::ffi::CString::new(tokenizer_path).expect("Failed to convert path to CString");

        unsafe {
            llamac::build_tokenizer(
                &mut tokenizer as *mut _,
                tokenizer_path.as_ptr() as *mut _,
                vocab_size,
            );
        }

        Tokenizer { inner: tokenizer }
    }

    fn encode(&mut self, text: &str) -> Vec<i32> {
        let c_text = std::ffi::CString::new(text).expect("Failed to convert text to CString");
        let mut prompt_tokens: Vec<i32> = Vec::with_capacity(text.len() + 3); // +3 for '\0', ?BOS, ?EOS
        let mut num_prompt_tokens: i32 = 0;

        unsafe {
            llamac::encode(
                &mut self.inner as *mut _,
                c_text.as_ptr() as *mut _,
                1,
                0,
                prompt_tokens.as_mut_ptr(),
                &mut num_prompt_tokens as *mut _,
            );

            prompt_tokens.set_len(num_prompt_tokens as usize);
        }

        prompt_tokens
    }
    // char* decode(Tokenizer* t, int prev_token, int token);

    fn decode(&mut self, prev_token: i32, token: i32) -> String {
        let ptr = unsafe { llamac::decode(&mut self.inner as *mut _, prev_token, token) };
        unsafe { std::ffi::CStr::from_ptr(ptr).to_string_lossy().to_string() }
    }
}

impl Drop for Tokenizer {
    fn drop(&mut self) {
        unsafe {
            llamac::free_tokenizer(&mut self.inner as *mut _);
        }
    }
}

struct Sampler {
    inner: llamac::Sampler,
}

impl Sampler {
    fn new(vocab_size: i32, temperature: f32, topp: f32, rng_seed: u64) -> Self {
        let mut sampler: llamac::Sampler = unsafe { std::mem::zeroed() };

        unsafe {
            llamac::build_sampler(
                &mut sampler as *mut _,
                vocab_size,
                temperature,
                topp,
                rng_seed,
            );
        }

        Sampler { inner: sampler }
    }

    fn sample(&mut self, logits: &[f32]) -> i32 {
        unsafe { llamac::sample(&mut self.inner as *mut _, logits.as_ptr() as *mut _) }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            llamac::free_sampler(&mut self.inner as *mut _);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_speaks() {
        // Create a new TransformerSystem instance
        let mut transformer_system =
            TransformerSystem::new("stories15M.bin", "llama2-c/tokenizer.bin", 1.0, 0.8, None);

        let token_count = transformer_system.generate(
            "Alice is walking into a test.",
            /* steps */ 10,
            |x| println!("{}", x),
        );

        assert_ne!(token_count, 0, "We should have generated something.");
    }
}
