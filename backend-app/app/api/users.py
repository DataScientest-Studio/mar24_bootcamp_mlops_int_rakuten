import json
from fastapi import Depends, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.sql_db import crud 
import app.models.database as models # INFO: Naming?
import app.models.user as api_m 
from app.sql_db.crud import get_db
from app.sql_db.database import engine
from app.api.auth import get_current_active_user
# WARNING: only temporary not needed if connected to already existing database
models.Base.metadata.create_all(bind=engine)


router = APIRouter(
    prefix='/users',
    tags=['user'],
    # dependencies=[Depends(get_current_active_user)]
)

# TODO: change endpoint?
# TODO: check and change respons models! 

@router.post("/", response_model=api_m.User)
def create_user(user: api_m.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)


@router.get("/users/", response_model=list[api_m.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@router.get("/id/{user_id}", response_model=api_m.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail=f"{db_user} User not found")
    return db_user

@router.get("/mail/{user_mail}", response_model=api_m.User, tags=['user'])
def read_user_by_mail(user_mail: str, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user_mail)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.put('/{user_id}/update_admin/', response_model=api_m.User)
def promote_to_admin(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail=f"User not found")
    #TODO: improve exception handeling
    if db_user.is_admin is None:
        updated_user = crud.update_is_admin(db, db_user)
    else:
        raise HTTPException(status_code=404, detail=f"User is already admin")
    return 

@router.put('/{user_id}/update_state/')
def promote_to_admin(user_id: int,new_state:bool, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail=f"User not found")
    #TODO: improve exception handeling
    if db_user.is_active == new_state:
        raise HTTPException(status_code=404, detail=f"User is_active is set to {new_state} already!")
    updated_user = crud.update_is_active(db, db_user, new_state)
    return JSONResponse(content={"Message": "User status updated"})
