from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

def validate_inputs(model: Optional[Type[BaseModel]], raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Returns (validated_dict, error_message). If model is None, returns raw as-is.
    """
    if model is None:
        return raw, None
    try:
        obj = model.model_validate(raw)
        return obj.model_dump(), None
    except ValidationError as e:
        return {}, str(e)
