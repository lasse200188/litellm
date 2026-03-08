"""
CRUD endpoints for storing reusable credentials.
"""

from typing import Any, Dict, List, Optional, cast

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, Path

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import HEALTH_CHECK_TIMEOUT_SECONDS
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.litellm_logging import _get_masked_values
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.health_check import (
    _update_litellm_params_for_health_check,
    run_with_timeout,
)
from litellm.proxy.management_endpoints.model_management_endpoints import (
    _add_model_to_db,
    get_db_model,
    update_db_model,
)
from litellm.proxy.utils import handle_exception_on_proxy, jsonify_object
from litellm.types.router import Deployment, LiteLLM_Params, updateDeployment
from litellm.types.utils import CreateCredentialItem, CredentialItem

router = APIRouter()


class CredentialHelperUtils:
    @staticmethod
    def encrypt_credential_values(credential: CredentialItem, new_encryption_key: Optional[str] = None) -> CredentialItem:
        """Encrypt values in credential.credential_values and add to DB"""
        encrypted_credential_values = {}
        for key, value in (credential.credential_values or {}).items():
            encrypted_credential_values[key] = encrypt_value_helper(value, new_encryption_key)

        # Return a new object to avoid mutating the caller's credential, which
        # is kept in memory and should remain unencrypted.
        return CredentialItem(
            credential_name=credential.credential_name,
            credential_values=encrypted_credential_values,
            credential_info=credential.credential_info or {},
        )


OPENROUTER_PROVIDER = "openrouter"
OPENROUTER_ACCESS_GROUP = "Openrouter"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_MODEL_NAME_PREFIX = "openrouter"


def _is_openrouter_provider(provider_name: Optional[str]) -> bool:
    return (provider_name or "").strip().lower() == OPENROUTER_PROVIDER


def _extract_openrouter_api_key(credential: CredentialItem) -> Optional[str]:
    api_key = (credential.credential_values or {}).get("api_key")
    if isinstance(api_key, str):
        api_key = api_key.strip()
    return api_key or None


def _openrouter_supports_reasoning(model_data: Dict[str, Any]) -> bool:
    supported_parameters = model_data.get("supported_parameters")
    if not isinstance(supported_parameters, list):
        return False
    supported_parameter_set = {
        str(item).strip().lower() for item in supported_parameters if item
    }
    reasoning_flags = {"reasoning", "include_reasoning", "reasoning_effort"}
    return len(reasoning_flags.intersection(supported_parameter_set)) > 0


def _build_openrouter_deployment(
    model_id: str,
    credential_name: str,
    supports_reasoning: bool,
) -> Deployment:
    openrouter_model_identifier = f"{OPENROUTER_MODEL_NAME_PREFIX}/{model_id}"
    return Deployment(
        model_name=openrouter_model_identifier,
        litellm_params=LiteLLM_Params(
            model=openrouter_model_identifier,
            litellm_credential_name=credential_name,
        ),
        model_info={
            "access_groups": [OPENROUTER_ACCESS_GROUP],
            "supports_reasoning": supports_reasoning,
        },
    )


async def _fetch_openrouter_models(api_key: str) -> List[Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(OPENROUTER_MODELS_URL, headers=headers)

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Failed to fetch OpenRouter models. HTTP {response.status_code}",
        )

    response_json = response.json()
    model_items = response_json.get("data", [])
    if not isinstance(model_items, list):
        raise HTTPException(
            status_code=502,
            detail="Unexpected OpenRouter /models response format.",
        )

    return [item for item in model_items if isinstance(item, dict)]


async def _run_model_health_check(litellm_params: Dict[str, Any]) -> Optional[str]:
    try:
        healthcheck_params = _update_litellm_params_for_health_check(
            model_info={},
            litellm_params=litellm_params.copy(),
        )
        result = await run_with_timeout(
            litellm.ahealth_check(
                model_params=healthcheck_params,
                mode="chat",
                prompt="test from litellm credential import",
                input=["test from litellm credential import"],
            ),
            HEALTH_CHECK_TIMEOUT_SECONDS,
        )
        if "error" in result:
            return str(result.get("error"))
        return None
    except Exception as e:
        return str(e)


@router.post(
    "/credentials",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def create_credential(
    request: Request,
    fastapi_response: Response,
    credential: CreateCredentialItem,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    Stores credential in DB.
    Reloads credentials in memory.
    """
    from litellm.proxy.proxy_server import llm_router, prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        if credential.model_id:
            if llm_router is None:
                raise HTTPException(
                    status_code=500,
                    detail="LLM router not found. Please ensure you have a valid router instance.",
                )
            # get model from router
            model = llm_router.get_deployment(credential.model_id)
            if model is None:
                raise HTTPException(status_code=404, detail="Model not found")
            credential_values = llm_router.get_deployment_credentials(
                credential.model_id
            )
            if credential_values is None:
                raise HTTPException(status_code=404, detail="Model not found")
            credential.credential_values = credential_values

        if credential.credential_values is None:
            raise HTTPException(
                status_code=400,
                detail="Credential values are required. Unable to infer credential values from model ID.",
            )
        processed_credential = CredentialItem(
            credential_name=credential.credential_name,
            credential_values=credential.credential_values,
            credential_info=credential.credential_info,
        )
        encrypted_credential = CredentialHelperUtils.encrypt_credential_values(
            processed_credential
        )
        credentials_dict = encrypted_credential.model_dump()
        credentials_dict_jsonified = jsonify_object(credentials_dict)
        await prisma_client.db.litellm_credentialstable.create(
            data={
                **credentials_dict_jsonified,
                "created_by": user_api_key_dict.user_id,
                "updated_by": user_api_key_dict.user_id,
            }
        )

        ## ADD TO LITELLM ##
        CredentialAccessor.upsert_credentials([processed_credential])

        return {"success": True, "message": "Credential created successfully"}
    except Exception as e:
        verbose_proxy_logger.exception(e)
        raise handle_exception_on_proxy(e)


@router.get(
    "/credentials",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def get_credentials(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    try:
        masked_credentials = [
            {
                "credential_name": credential.credential_name,
                "credential_values": _get_masked_values(credential.credential_values),
                "credential_info": credential.credential_info,
            }
            for credential in litellm.credential_list
        ]
        return {"success": True, "credentials": masked_credentials}
    except Exception as e:
        return handle_exception_on_proxy(e)


@router.get(
    "/credentials/by_name/{credential_name:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=CredentialItem,
)
@router.get(
    "/credentials/by_model/{model_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=CredentialItem,
)
async def get_credential(
    request: Request,
    fastapi_response: Response,
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
    model_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    from litellm.proxy.proxy_server import llm_router

    try:
        if model_id:
            if llm_router is None:
                raise HTTPException(status_code=500, detail="LLM router not found")
            model = llm_router.get_deployment(model_id)
            if model is None:
                raise HTTPException(status_code=404, detail="Model not found")
            credential_values = llm_router.get_deployment_credentials(model_id)
            if credential_values is None:
                raise HTTPException(status_code=404, detail="Model not found")
            masked_credential_values = _get_masked_values(
                credential_values,
                unmasked_length=4,
                number_of_asterisks=4,
            )
            credential = CredentialItem(
                credential_name="{}-credential-{}".format(model.model_name, model_id),
                credential_values=masked_credential_values,
                credential_info={},
            )
            # return credential object
            return credential
        elif credential_name:
            for credential in litellm.credential_list:
                if credential.credential_name == credential_name:
                    masked_credential = CredentialItem(
                        credential_name=credential.credential_name,
                        credential_values=_get_masked_values(
                            credential.credential_values,
                            unmasked_length=4,
                            number_of_asterisks=4,
                        ),
                        credential_info=credential.credential_info,
                    )
                    return masked_credential
            raise HTTPException(
                status_code=404,
                detail="Credential not found. Got credential name: " + credential_name,
            )
        else:
            raise HTTPException(
                status_code=404, detail="Credential name or model ID required"
            )
    except Exception as e:
        verbose_proxy_logger.exception(e)
        raise handle_exception_on_proxy(e)


@router.post(
    "/credentials/{credential_name:path}/import_models",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def import_models_from_credential(
    request: Request,
    fastapi_response: Response,
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Import all models for a credential provider.

    Currently only supports OpenRouter credentials.
    """
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_config,
        proxy_logging_obj,
        store_model_in_db,
    )

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        if store_model_in_db is not True:
            raise HTTPException(
                status_code=400,
                detail="Set `STORE_MODEL_IN_DB='True'` to import models.",
            )

        credential = next(
            (
                c
                for c in litellm.credential_list
                if c.credential_name == credential_name
            ),
            None,
        )
        if credential is None:
            raise HTTPException(status_code=404, detail="Credential not found")

        provider_name = (credential.credential_info or {}).get(
            "custom_llm_provider", ""
        )
        if not _is_openrouter_provider(cast(Optional[str], provider_name)):
            raise HTTPException(
                status_code=400,
                detail="Model import is currently only supported for Openrouter credentials.",
            )

        api_key = _extract_openrouter_api_key(credential)
        if api_key is None:
            raise HTTPException(
                status_code=400,
                detail="Openrouter credential is missing `api_key`.",
            )

        openrouter_models = await _fetch_openrouter_models(api_key=api_key)
        created_models: List[str] = []
        updated_models: List[str] = []
        failed_create: List[Dict[str, str]] = []
        failed_tests: List[Dict[str, str]] = []
        had_model_mutation = False

        for model_data in openrouter_models:
            model_id = model_data.get("id")
            if not isinstance(model_id, str) or model_id.strip() == "":
                continue

            supports_reasoning = _openrouter_supports_reasoning(model_data=model_data)
            deployment = _build_openrouter_deployment(
                model_id=model_id,
                credential_name=credential_name,
                supports_reasoning=supports_reasoning,
            )

            try:
                existing_model = await prisma_client.db.litellm_proxymodeltable.find_first(
                    where={"model_name": deployment.model_name}
                )
                if existing_model is None:
                    await _add_model_to_db(
                        model_params=deployment,
                        user_api_key_dict=user_api_key_dict,
                        prisma_client=prisma_client,
                    )
                    created_models.append(deployment.model_name)
                else:
                    existing_model_id = cast(Optional[str], existing_model.model_id)
                    if existing_model_id is None:
                        raise HTTPException(
                            status_code=500,
                            detail="Existing model row is missing model_id.",
                        )
                    db_model = await get_db_model(
                        model_id=existing_model_id,
                        prisma_client=prisma_client,
                    )
                    if db_model is None:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to load existing model '{existing_model_id}' for update.",
                        )
                    patch_data = updateDeployment(
                        model_name=deployment.model_name,
                        litellm_params=deployment.litellm_params,
                        model_info=deployment.model_info,
                    )
                    update_data = update_db_model(db_model, patch_data)
                    await prisma_client.db.litellm_proxymodeltable.update(
                        where={"model_id": existing_model_id},
                        data={
                            **update_data,
                            "updated_by": user_api_key_dict.user_id
                            or "litellm_proxy_admin",
                        },
                    )
                    updated_models.append(deployment.model_name)
                had_model_mutation = True

                healthcheck_error = await _run_model_health_check(
                    litellm_params=deployment.litellm_params.model_dump(
                        exclude_none=True
                    )
                )
                if healthcheck_error is not None:
                    failed_test = {
                        "model_name": deployment.model_name,
                        "error": healthcheck_error,
                    }
                    failed_tests.append(failed_test)
                    verbose_proxy_logger.warning(
                        "IMPORT_TEST_FAILED provider=%s credential=%s model_name=%s error=%s",
                        OPENROUTER_PROVIDER,
                        credential_name,
                        deployment.model_name,
                        healthcheck_error,
                    )
            except Exception as e:
                failure_detail = {"model_name": deployment.model_name, "error": str(e)}
                failed_create.append(failure_detail)
                verbose_proxy_logger.exception(
                    "OPENROUTER_IMPORT_MODEL_FAILED provider=%s credential=%s model_name=%s error=%s",
                    OPENROUTER_PROVIDER,
                    credential_name,
                    deployment.model_name,
                    str(e),
                )

        if had_model_mutation:
            await proxy_config.add_deployment(
                prisma_client=prisma_client,
                proxy_logging_obj=proxy_logging_obj,
            )

        return {
            "success": True,
            "provider": OPENROUTER_ACCESS_GROUP,
            "credential_name": credential_name,
            "counts": {
                "created": len(created_models),
                "updated": len(updated_models),
                "failed_tests": len(failed_tests),
                "failed_create": len(failed_create),
            },
            "created_models": created_models,
            "updated_models": updated_models,
            "failed_tests": failed_tests,
            "failed_create": failed_create,
        }
    except Exception as e:
        verbose_proxy_logger.exception(e)
        raise handle_exception_on_proxy(e)


@router.delete(
    "/credentials/{credential_name:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def delete_credential(
    request: Request,
    fastapi_response: Response,
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        await prisma_client.db.litellm_credentialstable.delete(
            where={"credential_name": credential_name}
        )

        ## DELETE FROM LITELLM ##
        litellm.credential_list = [
            cred
            for cred in litellm.credential_list
            if cred.credential_name != credential_name
        ]
        return {"success": True, "message": "Credential deleted successfully"}
    except Exception as e:
        return handle_exception_on_proxy(e)


def update_db_credential(
    db_credential: CredentialItem, updated_patch: CredentialItem, new_encryption_key: Optional[str] = None
) -> CredentialItem:
    """
    Update a credential in the DB.
    """
    merged_credential = CredentialItem(
        credential_name=db_credential.credential_name,
        credential_info=db_credential.credential_info,
        credential_values=db_credential.credential_values,
    )

    encrypted_credential = CredentialHelperUtils.encrypt_credential_values(
        updated_patch,
        new_encryption_key,
    )
    # update model name
    if encrypted_credential.credential_name:
        merged_credential.credential_name = encrypted_credential.credential_name

    # update litellm params
    if encrypted_credential.credential_values:
        # Encrypt any sensitive values
        encrypted_params = {
            k: v for k, v in encrypted_credential.credential_values.items()
        }

        merged_credential.credential_values.update(encrypted_params)

    # update model info
    if encrypted_credential.credential_info:
        """Update credential info"""
        if "credential_info" not in merged_credential.credential_info:
            merged_credential.credential_info = {}
        merged_credential.credential_info.update(encrypted_credential.credential_info)

    return merged_credential


@router.patch(
    "/credentials/{credential_name:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def update_credential(
    request: Request,
    fastapi_response: Response,
    credential: CredentialItem,
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        db_credential = await prisma_client.db.litellm_credentialstable.find_unique(
            where={"credential_name": credential_name},
        )
        if db_credential is None:
            raise HTTPException(status_code=404, detail="Credential not found in DB.")
        merged_credential = update_db_credential(db_credential, credential)
        credential_object_jsonified = jsonify_object(merged_credential.model_dump())
        await prisma_client.db.litellm_credentialstable.update(
            where={"credential_name": credential_name},
            data={
                **credential_object_jsonified,
                "updated_by": user_api_key_dict.user_id,
            },
        )
        return {"success": True, "message": "Credential updated successfully"}
    except Exception as e:
        return handle_exception_on_proxy(e)
