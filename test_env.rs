fn main() {
    let cuda = std::env::var("CARGO_FEATURE_CUDA").unwrap_or_default();
    println!("{}", cuda);
}
