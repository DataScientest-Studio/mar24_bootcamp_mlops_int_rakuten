from pydantic import BaseModel
from typing import Optional


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str

# TODO: Needed?
# class UserInDB(UserBase):
#     hashed_password: str

class User(UserBase):
    id: int
    is_active: bool
    is_admin: Optional[bool] = None

    class ConfigDict:
        from_attributes = True
