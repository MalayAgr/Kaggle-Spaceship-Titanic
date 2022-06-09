
from __future__ import annotations

from typing import Type


class BaseModel:
    name = None
    REGISTRY: dict[str, Type[BaseModel]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        if (name := cls.name) is not None:
            BaseModel.REGISTRY[name] = cls



class BaseOptunaCVModel(BaseModel):
    pass
