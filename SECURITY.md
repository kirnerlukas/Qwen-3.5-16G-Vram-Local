# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

This is a configuration and documentation repository. If you discover a security vulnerability in:

1. **This repository's code** — Please open an issue or email the maintainer
2. **llama.cpp** — Report to [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
3. **Qwen models** — Report to [QwenLM/Qwen](https://github.com/QwenLM/Qwen)

## Best Practices

When running local LLM servers:

- ✅ Bind to `127.0.0.1` (localhost) only — never expose to public networks
- ✅ Use a firewall if you must expose to a local network
- ✅ Keep llama.cpp updated to the latest version
- ✅ Download models only from trusted sources (official HuggingFace repos)
- ❌ Never commit API keys or credentials to the repository
