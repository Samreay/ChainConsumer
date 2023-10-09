from typing import Any

from pydantic import BaseModel, ConfigDict


class BetterBase(BaseModel):
    _user_specified: set[str] = set()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._user_specified = set(kwargs.keys())

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_user_specified_dump(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._user_specified}
