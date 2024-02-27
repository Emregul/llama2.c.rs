use std::time::Instant;

use llama_rust::TransformerSystem;

fn main() {
    let start = Instant::now();

    let steps = 250;
    let mut transformer_system =
        TransformerSystem::new("stories15M.bin", "llama2-c/tokenizer.bin", 1.0, 0.8, None);

    let mut generated_output = String::new();

    let append_to_generated_output = |s: &str| {
        generated_output.push_str(s);
    };

    let token_count = transformer_system.generate("your_prompt", steps, append_to_generated_output);

    let duration = start.elapsed();

    println!("{}", generated_output);
    println!(
        "Time taken: {:?}, toks: {}, toks/sec {}",
        duration,
        token_count,
        (token_count as f64 / duration.as_millis() as f64) * 1000.0
    );
}
