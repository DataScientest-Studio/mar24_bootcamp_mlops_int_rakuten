from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
import app.models.database as models # INFO: Naming?
from app.sql_db.database import engine
from app.core.logger import log_requests
from .api import auth, users, predict_category

# WARNING: only temporary not needed if connected to already existing database
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(predict_category.router)

# @app.middleware("http")
# async def log_requests_middleware(request: Request, call_next):
#     request_body = await request.body()
#     response = await call_next(request)
#     # response_body = [chunk async for chunk in response.body_iterator]
#     # BUG: Leads to:
#     # raise LocalProtocolError("Too little data for declared Content-Length")
#     # response already consumed? 
#     response_body = ""
#     await log_requests(request, request_body, response, response_body)
#     return response

@app.get('/')
async def root():
    return {'message': 'live'}


