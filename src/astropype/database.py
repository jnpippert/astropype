"""TODO docstring"""

import operator as op
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from pandas import DataFrame

__all__ = [
    "create",
    "update",
    "show",
    "select_bias",
    "select_darks",
    "select_flats",
    "select_lights",
]

BLACKLIST = ["qhy_20240312163001.fits"]


def create(
    __name: Path = Path("qhy_database.pkl"), overwrite: bool = False, omit: bool = True
):
    """TODO docstring"""
    if isinstance(__name, str):
        __name = Path(__name)
    path = __name.cwd()
    if not overwrite and __name.name in os.listdir(path):
        if not omit:
            raise FileExistsError("File already exists. Use 'overwrite = true'.")
        return
    scheme = {
        "date": "datetime64[ns]",
        "filename": "str",
        "filepath": "str",
        "object": "str",
        "filter": "str",
        "exptime": "float",
        "tempavg": "float",
        "tempmin": "float",
        "tempmax": "float",
        "median": "float",
    }
    database = pd.DataFrame(columns=scheme.keys()).astype(scheme)
    database.to_pickle(Path(__name))


def update(__database: Path, __archivepath: Path):
    """TODO docstring"""
    dataframe = pd.read_pickle(__database)
    dates = dataframe["date"].drop_duplicates().values
    archive_dates = [pd.Timestamp(d.name) for d in __archivepath.iterdir() if d.is_dir() and d.name not in ["comments","logfiles"]]
    archive_dates.sort()
    for date in archive_dates:
        date = date.date().strftime("%Y%m%d")
        added_rows = 0
        if pd.Timestamp(date) in dates:
            print(f"[INFO] {date} already in database -> Skipping date.")
            continue
        print(f"[INFO] Adding observations from {date} to database.")
        for file in __archivepath.joinpath(date).iterdir():
            if not file.is_file():
                continue
            filename = file.name
            if not filename.endswith(".fits") or not filename.startswith("qhy"):
                continue
            if filename in BLACKLIST:
                print(f"[INFO] Omitting {filename}")
                continue
            try:
                obj = fits.getval(file, "OBJECT").lower()
            except:
                continue

            found_substring = False
            for substring in ["pm", "test", "foc", "hd", "toi", "tic", "hip"]:
                if substring in obj:
                    found_substring = True
                    break
            if found_substring:
                continue

            if "bias" in obj:
                if obj != "bias_pipeline":
                    continue
            if "dark" in obj:
                if obj != "dark_pipeline":
                    continue
            try:
                fil = fits.getval(file, "FILTER")
                if "slot" in fil.lower():
                    continue
            except:
                continue

            try:
                exp = fits.getval(file, "EXPTIME")
            except:
                continue
            try:
                tempavg = fits.getval(file, "TEMPAVG")
                tempmax = fits.getval(file, "TEMPMAX")
                tempmin = fits.getval(file, "TEMPMIN")
            except:
                continue

            if obj in ["sky", "bias_pipeline", "dark_pipeline"]:
                try:
                    median = np.median(fits.getdata(file)[2140:4282, 3200:6400])
                except:
                    median = 0
            else:
                median = 0
            print(filename, obj, fil, exp)
            dataframe = append(
                dataframe,
                {
                    "date": pd.Timestamp(date),
                    "filename": filename,
                    "filepath": str(file),
                    "object": obj,
                    "filter": fil,
                    "exptime": exp,
                    "tempavg": tempavg,
                    "tempmin": tempmin,
                    "tempmax": tempmax,
                    "median": median,
                },
            )
            added_rows += 1
        if added_rows == 0:
            dataframe = append(
                dataframe,
                {
                    "date": pd.Timestamp(date),
                    "filename": "None",
                    "filepath": "None",
                    "object": "None",
                    "filter": "None",
                    "exptime": "None",
                    "tempavg": -1,
                    "tempmin": -1,
                    "tempmax": -1,
                    "median": 0,
                },
            )
        dataframe.to_pickle(__database)


def append(__dataframe: DataFrame, row: dict) -> DataFrame:
    return __dataframe._append(row, ignore_index=True)


def select(__dataframe: DataFrame, column: str, value: Any, operator: str) -> DataFrame:
    operators = {
        "==": op.eq,
        "!=": op.ne,
        ">": op.gt,
        "<": op.lt,
        ">=": op.ge,
        "<=": op.le,
    }
    if operator not in operators:
        raise ValueError("Invalid operator")
    return __dataframe[operators[operator](__dataframe[column], value)]


def extract(__dataframe: DataFrame, column: str, distinct: bool = True) -> list:
    if distinct:
        return list(__dataframe[column].drop_duplicates().values)
    return list(__dataframe[column].values)


def show(__database: Path, __date: str = None) -> None:
    pd.set_option("display.max_rows", None)
    database = pd.read_pickle(__database)
    if __date is not None:
        database = select(database, "date", pd.Timestamp(__date), "==")
    print(database)


def select_flat_date_candidates(
    __database: Path,
    __filter: str,
    __date: str,
    __minnum: int = 5,
) -> DataFrame:
    dataframe = select(pd.read_pickle(__database), "object", "sky", "==")
    if pd.Timestamp(__date) >= pd.Timestamp("20240110"):
        print("[INFO] Considering dates only younger than 2024.01.10")
        dataframe = select(dataframe, "date", pd.Timestamp("20240110"), ">=")
    if pd.Timestamp(__date) <= pd.Timestamp("20231219"):
        print("[INFO] Considering dates only older than 2023.12.19")
        dataframe = select(dataframe, "date", pd.Timestamp("20231219"), "<=")
    if pd.Timestamp(__date) > pd.Timestamp("20231219") and pd.Timestamp(
        __date
    ) < pd.Timestamp("20240110"):
        print("[INFO] Considering dates between 2023.12.20 and 2024.01.09")
        dataframe = select(dataframe, "date", pd.Timestamp(20231219), ">")
        dataframe = select(dataframe, "date", pd.Timestamp(20240110), "<")
    dataframe = select(dataframe, "filter", __filter, "==")
    dataframe = select(dataframe, "median", 16500, ">=")
    dataframe = select(dataframe, "median", 23500, "<=")
    return dataframe.groupby("date").filter(lambda x: len(x) >= __minnum)


def collect_flat_frames(
    __dataframe: DataFrame, __date: str, __maxnum: int = 10
) -> tuple:
    date = pd.Timestamp(__date)
    dataframe = __dataframe["date"].drop_duplicates()
    date = __dataframe.loc[(abs((dataframe - date)).idxmin()), "date"]
    dataframe = select(__dataframe, "date", date, "==")
    dataframe = dataframe.sort_values("exptime", ascending=True)
    return list(dataframe["filepath"].values[:__maxnum]), datetime(
        date.year, date.month, date.day
    ).strftime("%Y%m%d")


def select_bias(__database: Path, __minnum: int = 5, __maxnum: int = 20):
    dataframe = select(pd.read_pickle(__database), "object", "bias_pipeline", "==")
    dataframe = select(dataframe, "exptime", 0.0, "==")
    dataframe = select(dataframe, "filter", "O", "==")
    dataframe = select(dataframe, "median", 0.0, "!=")
    dataframe = dataframe.sort_values("tempavg", ascending=True)
    if len(dataframe) > __maxnum:
        return list(dataframe["filepath"].values[:__maxnum])
    if len(dataframe) < __minnum:
        return []
    return list(dataframe["filepath"].values)


def select_darks(
    __database: Path,
    __exptime: float,
    __minnum: int = 5,
    __maxnum: int = 20,
):
    dataframe = select(pd.read_pickle(__database), "object", "dark_pipeline", "==")
    dataframe = select(dataframe, "exptime", float(__exptime), "==")
    dataframe = select(dataframe, "median", 0.0, "!=")
    dataframe = select(dataframe, "filter", "O", "==")
    dataframe = dataframe.sort_values("tempavg", ascending=True)
    if len(dataframe) > __maxnum:
        return list(dataframe["filepath"].values[:__maxnum])
    if len(dataframe) < __minnum:
        return []
    return list(dataframe["filepath"].values)


def select_flats(__database: Path, __filter: str, __date: str, __minnum: int = 5):
    dataframe = select_flat_date_candidates(__database, __filter, __date, __minnum)
    return collect_flat_frames(dataframe, __date)


def select_lights(
    __database: Path, __object: str, __filter: str, __date: str, __exptime: float
):
    dataframe = select(pd.read_pickle(__database), "object", __object.lower(), "==")
    dataframe = select(dataframe, "date", pd.Timestamp(__date), "==")
    dataframe = select(dataframe, "filter", __filter, "==")
    dataframe = select(dataframe, "exptime", float(__exptime), "==")
    return list(dataframe["filepath"].values)


def sort(__database: Path, __columns: list, __ascending: list):
    dataframe: DataFrame = pd.read_pickle(__database)
    dataframe = dataframe.sort_values(__columns, ascending=__ascending)
    dataframe.to_pickle(__database)


def show_observed_objects(
    __database: Path, __object: str = None, __filter: str = None, __date: str = None
):
    pd.set_option("display.max_rows", None)
    df = pd.read_pickle(__database)
    df = df[["date", "object", "exptime", "filter", "median"]]
    df = select(df, "object", "None", "!=")
    if isinstance(__date, str):
        df = select(df, "date", pd.Timestamp(__date), "==")
    if isinstance(__object, str):
        __object = __object.lower()
        __object = __object.split("_")[0]
        df = df[df["object"].str.contains(__object) == True]
    if isinstance(__filter, str):
        df = select(df, "filter", __filter, "==")
    df = (
        df.groupby(["object", "exptime", "filter"])["exptime"]
        .count()
        .reset_index(name="count")
    )
    df = df.sort_values(["object", "filter", "count"], ascending=[True, True, True])
    print(df)


def get_observed_dates(__database: Path, __object: str, __filter: str, __exptime: str):
    df = pd.read_pickle(__database)
    df = df[["date", "object", "exptime", "filter"]]
    df = select(df, "object", __object.lower(), "==")
    df = select(df, "exptime", float(__exptime), "==")
    df = select(df, "filter", __filter, "==")
    dates = df["date"].drop_duplicates().values
    strdates = ""
    for date in dates:
        date = pd.Timestamp(date)
        strdates += datetime(date.year, date.month, date.day).strftime("%Y%m%d")
        strdates += ","
    print(strdates.removesuffix(","))
