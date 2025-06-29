from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
drowsie = False

class DrowsieStatus(BaseModel):
    drowsie: bool

@app.post("/set_drowsie_true", response_model=DrowsieStatus)
def set_drowsie_true():
    global drowsie
    drowsie = True
    return {"drowsie": drowsie}

@app.post("/set_drowsie_false", response_model=DrowsieStatus)
def set_drowsie_false():
    global drowsie
    drowsie = False
    return {"drowsie": drowsie}

@app.get("/get_drowsie", response_model=DrowsieStatus)
def get_drowsie():
    return {"drowsie": drowsie}