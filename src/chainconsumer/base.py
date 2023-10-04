from pydantic import BaseModel, ConfigDict


class BetterBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
