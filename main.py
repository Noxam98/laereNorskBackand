from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from db import init_db, get_user, create_user
from g4f.client import AsyncClient
from fastapi.middleware.cors import CORSMiddleware
import json
import re
import logging
from logging.handlers import RotatingFileHandler
from task import task

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT configuration
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for authorization
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models
class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class PromptRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"
    web_search: bool = False

# Initialize AsyncClient for G4F
client = AsyncClient()

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Rotating file handler
file_handler = RotatingFileHandler('app.log', maxBytes=5 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# Utility functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# Event handler for startup
@app.on_event("startup")
async def startup():
    await init_db()


# API Endpoints
@app.post("/register")
async def register(user: UserRegister):
    if not user.username.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username cannot be empty"
        )
    if len(user.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long"
        )
    existing_user = await get_user(user.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    hashed_password = hash_password(user.password)
    return await create_user(user.username, hashed_password)


@app.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    user = await get_user(credentials.username)
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access_token = create_token({"sub": credentials.username}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    refresh_token = create_token({"sub": credentials.username}, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))

    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}


@app.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user = await get_user(username)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

        new_access_token = create_token({"sub": username}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        new_refresh_token = create_token({"sub": username}, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))

        return {"access_token": new_access_token, "refresh_token": new_refresh_token, "token_type": "bearer"}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")


@app.get("/protected")
async def protected_route(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"message": "Access granted", "user": payload["sub"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


@app.get("/generate/")
async def generate_response(prompt: str, model: str = "deepseek-r1"):
    """
    Generates a response from an AI model based on the provided prompt.

    Parameters:
    - prompt: The text prompt for the AI. (required)
    - model: The AI model to use for generation. (default: deepseek-r1)

    Returns:
    - A JSON response containing the AI's answer and processing time.
    """
    try:
        logger.info(f"Received request: prompt='{prompt}', model='{model}'")
        start_time = datetime.now()
        prompt_with_task = [
            {
                "role": "system",
                "content": task
            },
            {
                "role": "user",
                "content": f"Текст запроса от пользователя: >>{prompt}<<"
            }
        ]
        print(prompt_with_task)
        response = await client.chat.completions.create(
            model=model,
            messages=prompt_with_task
        )
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Response received in {processing_time:.2f} seconds")

        if response.choices and response.choices[0].message.content:
            logger.debug(f"Model response: {response.choices[0].message.content}")
            match = re.search(r'```json\n(.*?)\n```', response.choices[0].message.content, re.DOTALL)
            if match:
                json_str = match.group(1)
                print(json_str)
                try:
                    data = json.loads(json_str)
                    return {
                        "status": "success",
                        "response": data,
                        "processing_time": processing_time
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decoding error: {str(e)}")
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to decode JSON response")
            else:
                logger.warning("No JSON found in the model's response")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No JSON found in the response")
        else:
            logger.error("Empty response from the model")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Empty response from the model")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.2", port=8000)