from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from .api import auth, users, predict_category

app = FastAPI()

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(predict_category.router)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get('/')
async def root():
    return {'message': 'live'}


