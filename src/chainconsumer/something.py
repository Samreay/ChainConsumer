from pydantic import BaseModel


class Something(BaseModel):
    message: str

    def get_message(self) -> str:
        """Gets the message.

        Returns:
            str: The message.
        """
        return self.message
