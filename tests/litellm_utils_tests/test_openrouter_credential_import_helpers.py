from litellm.proxy.credential_endpoints.endpoints import (
    _build_openrouter_deployment,
    _extract_openrouter_api_key,
    _is_openrouter_provider,
    _openrouter_supports_reasoning,
)
from litellm.types.utils import CredentialItem


def test_is_openrouter_provider_case_insensitive():
    assert _is_openrouter_provider("Openrouter") is True
    assert _is_openrouter_provider("openrouter") is True
    assert _is_openrouter_provider("Requesty") is False


def test_extract_openrouter_api_key_from_credential():
    credential = CredentialItem(
        credential_name="openrouter-cred",
        credential_values={"api_key": " sk-or-test "},
        credential_info={"custom_llm_provider": "Openrouter"},
    )

    assert _extract_openrouter_api_key(credential) == "sk-or-test"


def test_openrouter_reasoning_supported_when_supported_parameters_include_reasoning():
    model_data = {"id": "openai/gpt-5", "supported_parameters": ["tools", "reasoning"]}
    assert _openrouter_supports_reasoning(model_data) is True


def test_openrouter_reasoning_not_supported_when_parameter_absent():
    model_data = {"id": "openai/gpt-4.1", "supported_parameters": ["tools", "temperature"]}
    assert _openrouter_supports_reasoning(model_data) is False


def test_build_openrouter_deployment_sets_model_access_group_and_credential_name():
    deployment = _build_openrouter_deployment(
        model_id="openai/gpt-4.1",
        credential_name="openrouter-cred",
        supports_reasoning=True,
    )

    assert deployment.model_name == "openrouter/openai/gpt-4.1"
    assert deployment.litellm_params.model == "openrouter/openai/gpt-4.1"
    assert deployment.litellm_params.litellm_credential_name == "openrouter-cred"
    assert deployment.model_info.access_groups == ["Openrouter"]
    assert deployment.model_info.supports_reasoning is True
