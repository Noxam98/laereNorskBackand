from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from g4f.client import AsyncClient

from task import task

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники (для теста)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация клиента
client = AsyncClient()


class PromptRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"
    web_search: bool = False


@app.get("/generate/")
async def generate_response(prompt: str, model: str = "gpt-4o", web_search: bool = False):
    """
    Получение ответа от ИИ на GET-запрос с промтом

    Параметры:
    - prompt: текст запроса (обязательный)
    - model: используемая модель (по умолчанию gpt-4o-mini)
    - web_search: использовать ли веб-поиск (по умолчанию False)

    Возвращает:
    - JSON с ответом от ИИ
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": task + f"Слово для перевода от пользователя: {prompt}"
                }
            ],
            web_search=web_search
        )

        if response.choices:
            return {
                "status": "success",
                "response": response.choices[0].message.content,
                "model": model
            }
        else:
            raise HTTPException(status_code=500, detail="Пустой ответ от модели")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)