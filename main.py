import tempfile
import os
from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from predict.predict import LoadModel

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
async def predict(video: UploadFile = File(
    media_type=["video/mp4", "video/x-m4v", "video/*"]
)):
    try:
        print(video.filename)
        print(video.file)
        # print(video.read())
        # print(video.close())
        # print(video)
        with tempfile.TemporaryFile() as tmp_file:
            tmp_file.write(await video.read())
            # print(tmp_file.name)

            result = load_model.predict_v(tmp_file.name)

        return JSONResponse(
            content={
                "error": False, 
                "message": "Prediction successful!", 
                "data": str(result)}, 
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
    uvicorn.run(app, reload=True)