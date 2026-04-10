import math
import os
import sys
import json
import random
from contextlib import asynccontextmanager
from urllib import parse, request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.city_data import CITY_COORDINATES
from predict import predict_risk_score, train_prediction_bundle


model_bundle = None
ORS_API_KEY = os.getenv("ORS_API_KEY", "").strip()
ORS_ROUTE_URL = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"


class PredictionRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    cause_category: str
    cause_subcategory: str
    selected_city: str | None = None


class RoutePredictionRequest(BaseModel):
    start_latitude: float = Field(ge=-90, le=90)
    start_longitude: float = Field(ge=-180, le=180)
    end_latitude: float = Field(ge=-90, le=90)
    end_longitude: float = Field(ge=-180, le=180)
    cause_category: str
    cause_subcategory: str


def haversine_distance_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def resolve_nearest_city(latitude, longitude):
    best_city = None
    best_distance = None

    for city_name, (city_lat, city_lon) in CITY_COORDINATES.items():
        distance = haversine_distance_km(latitude, longitude, city_lat, city_lon)
        if best_distance is None or distance < best_distance:
            best_city = city_name
            best_distance = distance

    return best_city, best_distance


def get_risk_band(probability):
    if probability < 25:
        return "Low risk"
    if probability < 50:
        return "Medium risk"
    if probability < 75:
        return "High risk"
    return "Critical risk"


def interpolate_points(start_lat, start_lon, end_lat, end_lon, sample_count=7):
    if sample_count <= 1:
        return [(start_lat, start_lon)]

    points = []
    for index in range(sample_count):
        ratio = index / (sample_count - 1)
        points.append(
            (
                start_lat + (end_lat - start_lat) * ratio,
                start_lon + (end_lon - start_lon) * ratio,
            )
        )

    return points


def _build_seed(*values):
    seed = 2166136261
    for value in values:
        for character in f"{value:.6f}":
            seed ^= ord(character)
            seed = (seed * 16777619) & 0xFFFFFFFF
    return seed


def _point_on_line(start_point, end_point, ratio):
    return (
        start_point[0] + (end_point[0] - start_point[0]) * ratio,
        start_point[1] + (end_point[1] - start_point[1]) * ratio,
    )


def _offset_point(base_point, normal_vector, offset):
    return (
        base_point[0] + normal_vector[0] * offset,
        base_point[1] + normal_vector[1] * offset,
    )


def _sample_cubic_bezier(start_point, control_point_1, control_point_2, end_point, steps):
    samples = []
    for index in range(steps):
        t = index / max(steps - 1, 1)
        one_minus_t = 1 - t
        x = (
            (one_minus_t**3) * start_point[0]
            + 3 * (one_minus_t**2) * t * control_point_1[0]
            + 3 * one_minus_t * (t**2) * control_point_2[0]
            + (t**3) * end_point[0]
        )
        y = (
            (one_minus_t**3) * start_point[1]
            + 3 * (one_minus_t**2) * t * control_point_1[1]
            + 3 * one_minus_t * (t**2) * control_point_2[1]
            + (t**3) * end_point[1]
        )
        samples.append((x, y))
    return samples


def _catmull_rom_to_bezier(points, steps_per_segment):
    if len(points) < 2:
        return points

    samples = []
    for index in range(len(points) - 1):
        previous_point = points[index - 1] if index > 0 else points[index]
        start_point = points[index]
        end_point = points[index + 1]
        next_point = points[index + 2] if index + 2 < len(points) else points[index + 1]

        control_point_1 = (
            start_point[0] + (end_point[0] - previous_point[0]) / 6,
            start_point[1] + (end_point[1] - previous_point[1]) / 6,
        )
        control_point_2 = (
            end_point[0] - (next_point[0] - start_point[0]) / 6,
            end_point[1] - (next_point[1] - start_point[1]) / 6,
        )

        segment_samples = _sample_cubic_bezier(
            start_point,
            control_point_1,
            control_point_2,
            end_point,
            steps_per_segment,
        )
        if index > 0:
            segment_samples = segment_samples[1:]
        samples.extend(segment_samples)

    return samples


def _generate_smoothed_noise(length, seeded_random):
    values = [seeded_random.uniform(-1, 1) for _ in range(length)]
    kernel = [1, 4, 6, 4, 1]
    kernel_sum = sum(kernel)

    for _ in range(3):
        smoothed = []
        for index in range(length):
            weighted_value = 0
            for kernel_index, weight in enumerate(kernel):
                sample_index = min(max(index + kernel_index - 2, 0), length - 1)
                weighted_value += values[sample_index] * weight
            smoothed.append(weighted_value / kernel_sum)
        values = smoothed

    return values


def build_natural_route_geometry(start_lat, start_lon, end_lat, end_lon, sample_count=96):
    if sample_count <= 2:
        return [(start_lat, start_lon), (end_lat, end_lon)]

    average_latitude_radians = math.radians((start_lat + end_lat) / 2)
    longitude_scale = max(math.cos(average_latitude_radians), 0.35)

    start_point = (start_lon * longitude_scale, start_lat)
    end_point = (end_lon * longitude_scale, end_lat)
    delta_x = end_point[0] - start_point[0]
    delta_y = end_point[1] - start_point[1]
    planar_distance = math.hypot(delta_x, delta_y)
    if planar_distance == 0:
        return [(start_lat, start_lon)] * sample_count

    unit_normal = (-delta_y / planar_distance, delta_x / planar_distance)
    seeded_random = random.Random(_build_seed(start_lat, start_lon, end_lat, end_lon))

    primary_amplitude = min(max(planar_distance * 0.18, 0.08), 2.1)
    bump_one = primary_amplitude * seeded_random.uniform(0.72, 0.98)
    bump_two = primary_amplitude * seeded_random.uniform(0.68, 0.94)
    valley = primary_amplitude * seeded_random.uniform(-0.22, 0.16)

    anchor_definitions = [
        (0.0, 0.0),
        (0.22, bump_one),
        (0.5, valley),
        (0.78, bump_two),
        (1.0, 0.0),
    ]
    anchors = [
        _offset_point(_point_on_line(start_point, end_point, ratio), unit_normal, offset)
        for ratio, offset in anchor_definitions
    ]

    steps_per_segment = max(12, math.ceil(sample_count / (len(anchors) - 1)))
    curved_xy_points = _catmull_rom_to_bezier(anchors, steps_per_segment)
    noise_profile = _generate_smoothed_noise(len(curved_xy_points), seeded_random)

    curved_geometry = []
    for index, point in enumerate(curved_xy_points):
        ratio = index / max(len(curved_xy_points) - 1, 1)
        envelope = math.sin(math.pi * ratio) ** 1.25
        noisy_point = _offset_point(
            point,
            unit_normal,
            noise_profile[index] * primary_amplitude * 0.18 * envelope,
        )
        latitude = noisy_point[1]
        longitude = noisy_point[0] / longitude_scale
        curved_geometry.append((latitude, longitude))

    resampled_geometry = sample_points_along_geometry(curved_geometry, sample_count)
    if resampled_geometry[0] != (start_lat, start_lon):
        resampled_geometry[0] = (start_lat, start_lon)
    if resampled_geometry[-1] != (end_lat, end_lon):
        resampled_geometry[-1] = (end_lat, end_lon)

    return resampled_geometry


def fetch_ors_route_geometry(start_lat, start_lon, end_lat, end_lon):
    if not ORS_API_KEY:
        raise ValueError("ORS_API_KEY is not configured.")

    payload = json.dumps(
        {
            "coordinates": [
                [start_lon, start_lat],
                [end_lon, end_lat],
            ]
        }
    ).encode("utf-8")
    ors_request = request.Request(
        ORS_ROUTE_URL,
        data=payload,
        headers={
            "Authorization": ORS_API_KEY,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with request.urlopen(ors_request, timeout=12) as response:
        route_payload = json.loads(response.read().decode("utf-8"))

    features = route_payload.get("features", [])
    if not features:
        raise ValueError("No route returned by OpenRouteService.")

    feature = features[0]
    geometry = feature.get("geometry", {}).get("coordinates", [])
    if not geometry:
        raise ValueError("Route geometry missing from OpenRouteService response.")

    summary = feature.get("properties", {}).get("summary", {})
    lat_lon_geometry = [(coordinate[1], coordinate[0]) for coordinate in geometry]
    return {
        "provider": "openrouteservice",
        "distance_km": summary.get("distance", 0) / 1000,
        "duration_minutes": summary.get("duration", 0) / 60,
        "geometry": lat_lon_geometry,
    }


def fetch_osrm_route_geometry(start_lat, start_lon, end_lat, end_lon):
    url = OSRM_ROUTE_URL.format(
        start_lon=parse.quote(str(start_lon)),
        start_lat=parse.quote(str(start_lat)),
        end_lon=parse.quote(str(end_lon)),
        end_lat=parse.quote(str(end_lat)),
    )
    with request.urlopen(url, timeout=12) as response:
        payload = json.loads(response.read().decode("utf-8"))

    routes = payload.get("routes", [])
    if not routes:
        raise ValueError("No route returned by routing service.")

    route = routes[0]
    geometry = route.get("geometry", {}).get("coordinates", [])
    if not geometry:
        raise ValueError("Route geometry missing from routing response.")

    lat_lon_geometry = [(coordinate[1], coordinate[0]) for coordinate in geometry]
    return {
        "provider": "osrm",
        "distance_km": route.get("distance", 0) / 1000,
        "duration_minutes": route.get("duration", 0) / 60,
        "geometry": lat_lon_geometry,
    }


def fetch_route_geometry(start_lat, start_lon, end_lat, end_lon):
    route_errors = []

    if ORS_API_KEY:
        try:
            return fetch_ors_route_geometry(start_lat, start_lon, end_lat, end_lon)
        except Exception as exc:
            route_errors.append(f"OpenRouteService failed: {exc}")

    try:
        return fetch_osrm_route_geometry(start_lat, start_lon, end_lat, end_lon)
    except Exception as exc:
        route_errors.append(f"OSRM failed: {exc}")

    raise ValueError(" | ".join(route_errors) or "No routing providers available.")


def sample_points_along_geometry(geometry, sample_count=7):
    if not geometry:
        return []
    if len(geometry) <= sample_count:
        return geometry

    sampled_points = []
    last_index = len(geometry) - 1
    for sample_index in range(sample_count):
        position = round(sample_index * last_index / (sample_count - 1))
        sampled_points.append(geometry[position])

    deduplicated = []
    for point in sampled_points:
        if not deduplicated or deduplicated[-1] != point:
            deduplicated.append(point)

    return deduplicated


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle
    model_bundle = train_prediction_bundle()
    yield


app = FastAPI(title="Accident Risk API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    categories = model_bundle["category_to_subcategories"]
    cities = sorted(model_bundle["label_encoders"]["Million Plus Cities"].classes_.tolist())
    return {
        "model_name": model_bundle["model_name"],
        "supported_cities": cities,
        "category_to_subcategories": categories,
        "city_coordinates": {
            city: {"latitude": coords[0], "longitude": coords[1]}
            for city, coords in CITY_COORDINATES.items()
        },
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    supported_cities = set(model_bundle["label_encoders"]["Million Plus Cities"].classes_.tolist())
    selected_city = payload.selected_city.strip() if payload.selected_city else None

    if selected_city and selected_city in supported_cities:
        resolved_city = selected_city
        city_lat, city_lon = CITY_COORDINATES[resolved_city]
        nearest_distance_km = haversine_distance_km(
            payload.latitude,
            payload.longitude,
            city_lat,
            city_lon,
        )
        resolution_mode = "manual_city"
    else:
        resolved_city, nearest_distance_km = resolve_nearest_city(payload.latitude, payload.longitude)
        resolution_mode = "nearest_city"

    prediction = predict_risk_score(
        model_bundle,
        resolved_city,
        payload.cause_category,
        payload.cause_subcategory,
    )
    probability = prediction["risk_probability"]

    return {
        "input": payload.model_dump(),
        "resolved_city": resolved_city,
        "distance_to_city_km": round(nearest_distance_km, 2),
        "resolution_mode": resolution_mode,
        "risk_probability": round(probability, 2),
        "risk_band": get_risk_band(probability),
        "reference_stats": prediction["reference_stats"],
    }


@app.post("/predict-route")
def predict_route(payload: RoutePredictionRequest):
    route_source = "actual_route"
    route_notice = None

    try:
        route_data = fetch_route_geometry(
            payload.start_latitude,
            payload.start_longitude,
            payload.end_latitude,
            payload.end_longitude,
        )
        route_geometry = route_data["geometry"]
        route_distance_km = route_data["distance_km"]
        route_duration_minutes = route_data["duration_minutes"]
        route_provider = route_data["provider"]
        sample_points = sample_points_along_geometry(route_geometry)
    except Exception:
        route_source = "straight_line_fallback"
        route_notice = "Routing service unavailable, so a natural curved approximation was generated."
        route_geometry = build_natural_route_geometry(
            payload.start_latitude,
            payload.start_longitude,
            payload.end_latitude,
            payload.end_longitude,
            sample_count=96,
        )
        route_distance_km = haversine_distance_km(
            payload.start_latitude,
            payload.start_longitude,
            payload.end_latitude,
            payload.end_longitude,
        )
        route_duration_minutes = None
        route_provider = "fallback"
        sample_points = sample_points_along_geometry(route_geometry)

    if not sample_points:
        raise HTTPException(status_code=502, detail="Unable to derive route sample points.")

    samples = []
    for latitude, longitude in sample_points:
        resolved_city, nearest_distance_km = resolve_nearest_city(latitude, longitude)
        prediction = predict_risk_score(
            model_bundle,
            resolved_city,
            payload.cause_category,
            payload.cause_subcategory,
        )
        probability = prediction["risk_probability"]
        samples.append(
            {
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
                "resolved_city": resolved_city,
                "distance_to_city_km": round(nearest_distance_km, 2),
                "risk_probability": round(probability, 2),
                "risk_band": get_risk_band(probability),
                "reference_stats": prediction["reference_stats"],
            }
        )

    average_probability = sum(sample["risk_probability"] for sample in samples) / len(samples)
    highest_risk_sample = max(samples, key=lambda sample: sample["risk_probability"])
    unique_cities = list(dict.fromkeys(sample["resolved_city"] for sample in samples))

    return {
        "input": payload.model_dump(),
        "route_distance_km": round(route_distance_km, 2),
        "route_duration_minutes": round(route_duration_minutes, 2) if route_duration_minutes is not None else None,
        "route_source": route_source,
        "route_provider": route_provider,
        "route_notice": route_notice,
        "sample_count": len(samples),
        "average_risk_probability": round(average_probability, 2),
        "average_risk_band": get_risk_band(average_probability),
        "highest_risk_probability": highest_risk_sample["risk_probability"],
        "highest_risk_band": highest_risk_sample["risk_band"],
        "highest_risk_city": highest_risk_sample["resolved_city"],
        "cities_on_route": unique_cities,
        "route_geometry": [
            {"latitude": round(latitude, 5), "longitude": round(longitude, 5)}
            for latitude, longitude in route_geometry
        ],
        "samples": samples,
    }
