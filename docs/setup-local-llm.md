# Local LLM

NLWeb can run a language model completely offline by using the `transformers`
library. Place your model on disk and set the path in the `LOCAL_LLM_MODEL_PATH`
environment variable or update `code/config/config_llm.yaml` to reference the
path directly.

Example configuration snippet:

```yaml
endpoints:
  local:
    api_endpoint_env: LOCAL_LLM_MODEL_PATH
    llm_type: local
    models:
      high: local-model
      low: local-model
```

Ensure the required dependency is installed:

```sh
pip install transformers
```

Once configured you can select the `local` endpoint in `config_llm.yaml` or via
the `llm_provider` query parameter when running in development mode.
