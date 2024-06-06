import tempfile
import os
import asyncio
from dotenv import load_dotenv
from typing import Union
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from predict.predict import LoadModel

load_dotenv()
app = FastAPI()
load_model = LoadModel()

@app.get("/")
async def read_root():
    return JSONResponse(
        content={
            "error": False, 
            "message": "ML Prediction API is running!"}, 
        status_code=200)


@app.post("/predict")
def predict(video: UploadFile = File(
    media_type=["video/mp4", "video/x-m4v", "video/*"]
)):
    try:
        # print(video.filename)
        # print(video.file)
        # print(video.read())
        # print(video.close())
        # print(video)
        with tempfile.TemporaryFile() as tmp_file:
            tmp_file.write(video.file.read())
            # print(tmp_file.name)

            result = load_model.predict_v(tmp_file.name)

            tmp_file.close()

        return JSONResponse(
            content={
                "error": False, 
                "message": "Prediction successful!", 
                "data": result}, 
            status_code=200)
    except Exception as e:
        return JSONResponse(
            content={
                "error": True, 
                "message": "Prediction failed!", 
                "data": str(e)}, 
            status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=str(os.getenv('HOST')), 
        port=int(os.getenv('PORT')), 
        workers=int(os.getenv('WORKERS')),
        reload=True)