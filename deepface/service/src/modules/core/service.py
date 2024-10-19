# built-in dependencies
import pandas as pd
import traceback
from typing import Optional


from deepface.modules import verification


# project dependencies
from deepface import DeepFace

threshold = verification.find_threshold(model_name="VGG-Face", distance_metric="cosine")


# pylint: disable=broad-except


def represent(
    img_path: str,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
):
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        result["results"] = embedding_objs
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: str,
    img2_path: str,
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return obj
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400


def find(
        img_path: str,
        db_path: str,
        model_name: str,
        detector_backend: str,
        distance_metric: str,
        enforce_detection: bool,
        align: bool,
        anti_spoofing: bool,
        refresh_database: bool,
):
    try:
        dfs = DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
            refresh_database=refresh_database,
        )
        total = len(dfs[0])
        fields = ["identity", "threshold", "distance"]
        results = []
        for line in dfs:
            for i in range(total):
                if all(i < len(line.get(field)) for field in fields):  # Ensure index is within range for all fields
                    results.append({field: line.get(field)[i] for field in fields})
        return results
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400

def analyze(
    img_path: str,
    actions: list,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        result = {}
        demographies = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
            anti_spoofing=anti_spoofing,
        )
        result["results"] = demographies
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400
