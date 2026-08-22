fn main() {
    let item_str = "Ollama: qwen3.5 [Unknown Size]";
    let n = item_str
        .replace("Ollama:", "")
        .replace("Ollama Local:", "");
    let ollama_name = n
        .split('(').next().unwrap_or(&n)
        .split('[').next().unwrap_or(&n)
        .trim()
        .to_string();
    println!("{}", ollama_name);
}
