
from fastapi import FastAPI
import streamlit as st
import pandas as pd
import base64
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel


class Item(BaseModel):
    id: str
    value: str


app = FastAPI()
df = pd.read_csv("/home/ludo915/code/covsco/data/train/all_data_merged/fr/traindfknnimputed.csv")
date = (pd.to_datetime(df["date"].max()) + pd.Timedelta("1 Days")).strftime('%Y-%m-%d')

@app.get("/")
def index():
    return {"ok": date}

@app.get(
    "/pm25/",
    response_model=Item,
    responses={
        200: {
            "content": {"image/gif": {}},
            "description": "Return the JSON item or an image.",
        }
    },
)
async def read_item():
    filepath = "/home/ludo915/code/covsco/forecast/fr/PM2.5-concentration-" +date +".gif"
    return FileResponse(filepath, media_type="image/gif")

@app.get(
    "/co/",
    response_model=Item,
    responses={
        200: {
            "content": {"image/gif": {}},
            "description": "Return the JSON item or an image.",
        }
    },
)
async def read_item():
    filepath = "/home/ludo915/code/covsco/forecast/fr/CO-concentration-" +date +".gif"
    return FileResponse(filepath, media_type="image/gif")

@app.get(
    "/o3/",
    response_model=Item,
    responses={
        200: {
            "content": {"image/gif": {}},
            "description": "Return the JSON item or an image.",
        }
    },
)
async def read_item():
    filepath = "/home/ludo915/code/covsco/forecast/fr/O3-concentration-" +date +".gif"
    return FileResponse(filepath, media_type="image/gif")

@app.get(
    "/pm10/",
    response_model=Item,
    responses={
        200: {
            "content": {"image/gif": {}},
            "description": "Return the JSON item or an image.",
        }
    },
)
async def read_item():
    filepath = "/home/ludo915/code/covsco/forecast/fr/PM10-concentration-" +date +".gif"
    return FileResponse(filepath, media_type="image/gif")

@app.get(
    "/no2/",
    response_model=Item,
    responses={
        200: {
            "content": {"image/gif": {}},
            "description": "Return the JSON item or an image.",
        }
    },
)
async def read_item():
    filepath = "/home/ludo915/code/covsco/forecast/fr/NO2-concentration-" +date +".gif"
    return FileResponse(filepath, media_type="image/gif")

@app.get(
    "/so2/",
    response_model=Item,
    responses={
        200: {
            "content": {"image/gif": {}},
            "description": "Return the JSON item or an image.",
        }
    },
)
async def read_item():
    filepath = "/home/ludo915/code/covsco/forecast/fr/SO2-concentration-" +date +".gif"
    return FileResponse(filepath, media_type="image/gif")

@app.get(
    "/newhospi/",
    response_model=Item,
    responses={
        200: {
            "content": {"image/gif": {}},
            "description": "Return the JSON item or an image.",
        }
    },
)
async def read_item():
    filepath = "/home/ludo915/code/covsco/forecast/fr/newhospidepartementlevel-" +date +".gif"
    return FileResponse(filepath, media_type="image/gif")

    