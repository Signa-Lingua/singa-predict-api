import os
import tempfile

from dotenv import load_dotenv
# from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from predict.predict import Model
from service.gcloud_storage import get_model

load_dotenv()

app = FastAPI()

def model():
    model_path = get_model()
    load_model = Model(model_path)

    return load_model


@app.get("/")
async def read_root():
    return JSONResponse(
        content={"error": False, "message": "ML Prediction API is running!"},
        status_code=200,
    )


@app.post("/predict")
def predict(
    video: UploadFile = File(media_type=["video/mp4", "video/x-m4v", "video/*"])
):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(video.file.read())
            print(f"This is the temp file path: {tmp_file.name}")
            tmp_file.close()
            lm = model()
            result = lm.process_video(tmp_file.name)

        return JSONResponse(
            content={
                "error": False,
                "message": "Prediction successful!",
                "data": result,
            },
            status_code=200,
        )
    except Exception as e:
        return JSONResponse(
            content={"error": True, "message": "Prediction failed!", "data": str(e)},
            status_code=500,
        )
    finally:
        if os.path.exists(tmp_file.name):
            os.remove(tmp_file.name)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=str(os.getenv("HOST")),
        port=int(os.getenv("PORT")),
        reload=True,
    )
