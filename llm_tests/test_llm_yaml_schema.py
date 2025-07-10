# Copyright Axelera AI, 2025
import pytest

from axelera.app import yaml_parser


def get_llm_yaml_paths():
    # Use yaml_parser to get all LLM YAMLs, as in the old sanity test
    network_yaml_info = yaml_parser.get_network_yaml_info(
        include_collections=['llm_local', 'llm_cards']
    )
    all_infos = network_yaml_info.get_all_info()
    assert all_infos, "No LLM YAMLs found!"
    # Determine if each YAML is a model card or not
    result = []
    for info in all_infos:
        require_internal_model_card = "model_cards/llm" in info.yaml_path
        result.append((info.yaml_path, require_internal_model_card))
    return result


@pytest.mark.parametrize("yaml_path,require_internal_model_card", get_llm_yaml_paths())
def test_llm_yaml_schema(yaml_path, require_internal_model_card):
    from axelera.app.yaml_parser import model_from_path

    model = model_from_path(yaml_path)
    assert model is not None, f"Failed to load YAML: {yaml_path}"

    # Top-level keys
    required_keys = ["axelera-model-format", "name", "description", "models"]
    if require_internal_model_card:
        required_keys.append("internal-model-card")
    for key in required_keys:
        assert key in model, f"Missing top-level key '{key}' in {yaml_path}"

    # Only one model
    assert len(model["models"]) == 1, f"LLM YAML should have only one model: {yaml_path}"
    model_name = next(iter(model["models"].keys()))
    m = model["models"][model_name]

    # Model fields
    for field in (
        "precompiled_url",
        "precompiled_path",
        "precompiled_md5",
        "task_category",
        "extra_kwargs",
    ):
        assert field in m, f"Model missing '{field}': {yaml_path}"
    assert (
        m["task_category"] == "LanguageModel"
    ), f"task_category should be LanguageModel: {yaml_path}"
    assert m["precompiled_url"].startswith(
        "https://llm.axelera.ai"
    ), f"precompiled_url should start with https://llm.axelera.ai: {yaml_path}, got {m['precompiled_url']}"

    # Assert precompiled_path starts with 'build/<model_name>/'
    expected_prefix = f"build/{model_name}/"
    assert m["precompiled_path"].startswith(
        expected_prefix
    ), f"precompiled_path should start with '{expected_prefix}': {yaml_path}, got {m['precompiled_path']}"
    # NOTE: All model assets are zipped such that there is a 'model' directory inside the zip,
    # and all relevant files are placed within this 'model' directory. We do not test the contents
    # of the zip here, as the files are large and we want to avoid always downloading and checking them.

    # extra_kwargs.llm fields
    llm_kwargs = m.get("extra_kwargs", {}).get("llm", {})
    allowed_fields = {
        "model_name",
        "max_tokens",
        "embeddings_url",
        "embeddings_md5",
        "min_response_space",
        "tokenizer_url",
        "tokenizer_md5",
        "ddr_requirement_gb",
    }
    for field in ("model_name", "max_tokens", "embeddings_url", "embeddings_md5"):
        assert field in llm_kwargs, f"extra_kwargs.llm missing '{field}': {yaml_path}"
    # tokenizer_url is optional, but if present, must be a string and a valid URL
    if "tokenizer_url" in llm_kwargs:
        assert (
            "tokenizer_md5" in llm_kwargs
        ), f"extra_kwargs.llm missing 'tokenizer_md5': {yaml_path}"
        assert isinstance(
            llm_kwargs["tokenizer_url"], str
        ), f"tokenizer_url must be a string: {yaml_path}"
        assert llm_kwargs["tokenizer_url"].startswith(
            "http"
        ), f"tokenizer_url must be a valid URL: {yaml_path}"
    # Check for no extra fields
    extra_fields = set(llm_kwargs.keys()) - allowed_fields
    assert not extra_fields, f"extra_kwargs.llm has unexpected fields {extra_fields}: {yaml_path}"

    # internal-model-card fields (only for model cards)
    if require_internal_model_card:
        card = model["internal-model-card"]
        for field in ("card_name", "model_card", "production_ML_framework", "license"):
            assert field in card, f"internal-model-card missing '{field}': {yaml_path}"
        allowed_frameworks = {"PyTorch", "TensorFlow"}
        assert (
            card["production_ML_framework"] in allowed_frameworks
        ), f"production_ML_framework should be one of {allowed_frameworks}: {yaml_path}, got {card['production_ML_framework']}"
