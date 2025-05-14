#!/Users/owenleung/anaconda3/envs/GLH/bin/python
"""
Geolocation Data Processor
Processes geolocation data from JSON files, performs filtering, downsampling,
imputation, and clustering, then exports to CSV.
"""

import os
import json
import datetime
import argparse
import numpy as np
import pandas as pd
import osmnx as ox
from shapely.geometry import Point
from datetime import timedelta
from sklearn.cluster import KMeans
import math
from mobility_indices import calculate_mobility_indices
import osmnx as ox

gdf_hk = ox.geocode_to_gdf("Hong Kong SAR")
geom_hk = gdf_hk.loc[0, "geometry"]


## Process records


def simple_euclidean_distance_km(lat1, lon1, lat2, lon2):
    """Calculate Euclidean distance between two points in km."""
    if pd.isnull(lat1) or pd.isnull(lon1) or pd.isnull(lat2) or pd.isnull(lon2):
        return np.nan
    # Constants for Hong Kong region (latitude ~22 degrees)
    KM_PER_DEGREE_LATITUDE = 111
    KM_PER_DEGREE_LONGITUDE = 102

    # Calculate differences in coordinates
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Convert degree differences to kilometers
    delta_lat_km = delta_lat * KM_PER_DEGREE_LATITUDE
    delta_lon_km = delta_lon * KM_PER_DEGREE_LONGITUDE

    # Calculate Euclidean distance in kilometers
    distance_km = math.sqrt(delta_lat_km**2 + delta_lon_km**2)

    return distance_km


def format_datetime(dt_str, to_gmt8=False, return_string=True):
    """Format datetime with option to convert to GMT+8."""
    # Note: All timestamps are in Hong Kong time (GMT+8) but stored as timezone-naive datetimes
    if isinstance(dt_str, str):
        if dt_str.endswith("Z"):
            dt = datetime.datetime.fromisoformat(dt_str[:-1])
        else:
            dt = datetime.datetime.fromisoformat(dt_str)
    else:
        dt = dt_str

    if to_gmt8:
        dt = dt + datetime.timedelta(hours=8)

    if return_string:
        return dt.isoformat()
    else:
        return dt


def preprocess_records_json(input_file):
    """
    Preprocess Google Location History data from Records.json file.
    Converts raw data into the expected format for further processing.
    """
    with open(input_file) as f:
        records_json = json.load(f)

    coordinates = []
    for record in records_json.get("locations", []):
        if "accuracy" in record:
            try:
                new_coordinates = {
                    "latitude": record.get("latitudeE7") / 1e7,
                    "longitude": record.get("longitudeE7") / 1e7,
                    "accuracy": record.get("accuracy"),
                    "source": record.get("source"),
                    "devicetag": record.get("deviceTag"),
                    "timestamp": format_datetime(
                        record.get("timestampMs") or record.get("timestamp"),
                        return_string=True,
                    ),
                }
                coordinates.append(new_coordinates)
            except Exception as e:
                print(f"Error processing record: {e}")
                print(record)

    return coordinates


def filter_coords(
    coords,
    center_date,
    start_adjust_days,
    duration_days,
    accuracy_m=200,
    in_hk=True,
    to_gmt8=True,
    geom_hk=geom_hk,
):
    """Filter coordinates based on time, accuracy, and location."""
    # Convert timestamps to datetime if needed
    if to_gmt8:
        for coord in coords:
            coord["timestamp"] = format_datetime(
                coord["timestamp"], to_gmt8=True, return_string=False
            )

    # Filter by timestamp
    center_date = format_datetime(center_date, to_gmt8=False, return_string=False)
    start_time = center_date + datetime.timedelta(days=start_adjust_days)
    end_time = start_time + datetime.timedelta(days=duration_days)

    coords = [coord for coord in coords if start_time <= coord["timestamp"] <= end_time]

    # Filter by accuracy
    coords = [coord for coord in coords if 0 < coord["accuracy"] < accuracy_m]

    # Filter by location (Hong Kong)
    if in_hk and geom_hk is not None:
        coords = [
            coord
            for coord in coords
            if geom_hk.intersects(Point((coord["longitude"], coord["latitude"])))
        ]

    return coords


def filter_by_speed(df):
    """Remove points with unrealistic speeds (>100 km/h)."""

    def calculate_speeds(df):
        coords_diff = (
            df[["latitude", "longitude"]]
            .diff()
            .abs()
            .multiply([111, 102], axis="columns")
            .pow(2)
            .sum(axis=1)
            .pow(0.5)
        )
        time_diff_hours = df.index.to_series().diff().dt.total_seconds().div(3600)
        df["speed_per_hour"] = coords_diff.div(time_diff_hours).fillna(0)

    calculate_speeds(df)

    while True:
        excessive_speed_indices = df[df["speed_per_hour"] > 100].index
        if excessive_speed_indices.empty:
            break
        df = df.drop(excessive_speed_indices)
        calculate_speeds(df)

    return df.drop(columns=["speed_per_hour"])


def downsample_coordinates(df):
    """Step 2: Downsample coordinates to 5-minute intervals."""
    df["hour"] = df.index.floor("H")
    grouped = df.groupby("hour")

    downsampled_coords = []

    for name, group in grouped:
        lat_std = np.std(group["latitude"])
        lon_std = np.std(group["longitude"])

        if lat_std < 0.01 and lon_std < 0.01:
            hour_mean_x_12 = [
                {
                    "timestamp": name + timedelta(minutes=i * 5),
                    "latitude": np.mean(group["latitude"]),
                    "longitude": np.mean(group["longitude"]),
                    "accuracy": np.mean(group["accuracy"]),
                    "frequency": len(group) / 12,
                }
                for i in range(12)
            ]
            downsampled_coords.extend(hour_mean_x_12)
        else:
            median_or_empty_x_12 = [
                {
                    "timestamp": name + timedelta(minutes=i * 5),
                    "latitude": (
                        np.median(five_min_group["latitude"])
                        if not five_min_group.empty
                        else np.nan
                    ),
                    "longitude": (
                        np.median(five_min_group["longitude"])
                        if not five_min_group.empty
                        else np.nan
                    ),
                    "accuracy": (
                        np.median(five_min_group["accuracy"])
                        if not five_min_group.empty
                        else np.nan
                    ),
                    "frequency": len(five_min_group) if not five_min_group.empty else 0,
                }
                for i in range(12)
                for five_min_group in [
                    group[
                        (group.index >= name + timedelta(minutes=i * 5))
                        & (group.index < name + timedelta(minutes=(i + 1) * 5))
                    ]
                ]
            ]
            downsampled_coords.extend(median_or_empty_x_12)

    downsampled_df = pd.DataFrame(downsampled_coords)
    return downsampled_df.set_index("timestamp")


def impute_missing_data(df):
    """Step 3: Impute missing data points based on movement patterns."""
    resampled_df = df.resample("5T").mean()
    gaps = resampled_df.isnull().any(axis=1)
    gap_starts = gaps & (~gaps.shift(1, fill_value=False))
    gap_ends = gaps & (~gaps.shift(-1, fill_value=False))

    for start, end in zip(resampled_df[gap_starts].index, resampled_df[gap_ends].index):
        adjusted_start = start - pd.Timedelta(minutes=5) if start == end else start
        adjusted_end = end + pd.Timedelta(minutes=5) if start == end else end

        prev_valid_idx = resampled_df.loc[:adjusted_start].last_valid_index()
        next_valid_idx = resampled_df.loc[adjusted_end:].first_valid_index()

        if prev_valid_idx is not None and next_valid_idx is not None:
            prev_valid = resampled_df.loc[prev_valid_idx]
            next_valid = resampled_df.loc[next_valid_idx]

            if not pd.isnull(prev_valid["latitude"]) and not pd.isnull(
                next_valid["latitude"]
            ):
                prev_valid_hour = prev_valid_idx.hour
                time_difference = next_valid_idx - prev_valid_idx
                distance = simple_euclidean_distance_km(
                    prev_valid["latitude"],
                    prev_valid["longitude"],
                    next_valid["latitude"],
                    next_valid["longitude"],
                )

                if distance <= 500:
                    if (
                        21 <= prev_valid_hour < 24
                        and time_difference <= pd.Timedelta(hours=12)
                    ) or (time_difference <= pd.Timedelta(hours=2)):
                        mean_coord = (next_valid + prev_valid) / 2
                        resampled_df.loc[start:end] = resampled_df.loc[
                            start:end
                        ].fillna(mean_coord)

    return resampled_df


def calculate_distances(df):
    """Calculate distances between consecutive coordinates."""
    df["prev_latitude"] = df["latitude"].shift(1)
    df["prev_longitude"] = df["longitude"].shift(1)
    df["distance"] = df.apply(
        lambda row: simple_euclidean_distance_km(
            row["prev_latitude"],
            row["prev_longitude"],
            row["latitude"],
            row["longitude"],
        ),
        axis=1,
    )
    df["speed_km_h"] = df["distance"] * 12  # 12 = 60 min/hour ÷ 5 min intervals
    mask = df[["latitude", "longitude"]].isnull().any(axis=1)
    df.loc[mask, ["distance", "speed_km_h"]] = np.nan
    return df


def calculate_moving_averages_with_speed(
    df, window_central=3, window_backward=3, window_forward=3
):
    """Calculate moving averages for speed."""
    df["x_cen_speed"] = (
        df["speed_km_h"].rolling(window=window_central, center=True).mean()
    )
    df["x_back_speed"] = (
        df["speed_km_h"].rolling(window=window_backward, min_periods=1).mean()
    )
    df["x_fwd_speed"] = (
        df["speed_km_h"]
        .rolling(window=window_forward, min_periods=1)
        .mean()
        .shift(-(window_forward - 1))
    )
    return df


def identify_stationary_points(df):
    """Step 4: Identify stationary points based on speed thresholds."""
    imputed_df = df.copy()
    df_real_distances = calculate_distances(imputed_df)
    df_speed_moving_averages = calculate_moving_averages_with_speed(df_real_distances)

    threshold_speed = 1.5  # Threshold in km/h
    df_speed_moving_averages["transitioning"] = (
        (df_speed_moving_averages["x_cen_speed"] > threshold_speed)
        | (df_speed_moving_averages["x_back_speed"] > threshold_speed)
        | (df_speed_moving_averages["x_fwd_speed"] > threshold_speed)
    )
    df_speed_moving_averages["stationary"] = ~df_speed_moving_averages["transitioning"]
    df_speed_moving_averages.loc[
        df_speed_moving_averages["speed_km_h"].isnull(), ["transitioning", "stationary"]
    ] = np.nan
    return df_speed_moving_averages


def min_simple_euclidean_distance(centroids):
    """Calculate minimum distance between cluster centroids."""
    min_distance = np.inf
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distance = simple_euclidean_distance_km(
                centroids[i][0], centroids[i][1], centroids[j][0], centroids[j][1]
            )
            min_distance = min(min_distance, distance)
    return min_distance


def cluster_stationary_points(df):
    """Step 5: Cluster stationary points using KMeans."""
    stationary_df = df[df["stationary"] == True].copy()
    if stationary_df.empty:
        df["cluster_label"] = -1
        return df

    coordinates = stationary_df[["latitude", "longitude"]].values

    K = 2
    threshold_distance = 0.4
    while True:
        kmeans = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42).fit(
            coordinates
        )
        if min_simple_euclidean_distance(kmeans.cluster_centers_) < threshold_distance:
            break
        K += 1

    stationary_df["cluster_label"] = kmeans.labels_
    df["cluster_label"] = -1  # Default for non-stationary points
    df.loc[stationary_df.index, "cluster_label"] = stationary_df["cluster_label"]
    return df


def add_home_work_clusters(df):
    """Step 6: Add home and work clusters based on time patterns."""
    df = df.reset_index().rename(columns={"index": "timestamp"})

    home_time_df = df[df["timestamp"].dt.hour.isin(range(2, 7))]
    if not home_time_df.empty:
        assumed_home_clusters = (
            home_time_df.groupby(home_time_df["timestamp"].dt.date)["cluster_label"]
            .agg(
                lambda x: (
                    x[x != -1].mode()[0] if not x[x != -1].mode().empty else np.nan
                )
            )
            .reset_index()
        )
        assumed_home_clusters.columns = ["date", "assumed_home_cluster"]

        df["date"] = df["timestamp"].dt.date
        df = pd.merge(df, assumed_home_clusters, how="left", on="date")
    else:
        df["assumed_home_cluster"] = np.nan
        df["date"] = df["timestamp"].dt.date

    work_time_df = df[
        (
            (df["timestamp"].dt.hour.isin(range(10, 12)))
            | (df["timestamp"].dt.hour.isin(range(15, 17)))
        )
        & (df["timestamp"].dt.weekday.isin(range(5)))
    ]
    if not work_time_df.empty:
        assumed_work_clusters = (
            work_time_df.groupby(work_time_df["timestamp"].dt.date)["cluster_label"]
            .agg(
                lambda x: (
                    x[x != -1].mode()[0] if not x[x != -1].mode().empty else np.nan
                )
            )
            .reset_index()
        )
        assumed_work_clusters.columns = ["date", "assumed_work_cluster"]

        df = pd.merge(df, assumed_work_clusters, how="left", on="date")
    else:
        df["assumed_work_cluster"] = np.nan

    df = df.drop("date", axis=1)

    return df


def process_records(data_dir, svy_date):
    """Main process to filter coordinates, downsample, impute, cluster and export to CSV.

    Parameters:
    -----------
    data_dir : str
        Directory path containing a Records.json file with Google Location History records
    svy_date : datetime or str
        Date when the survey was completed, used as reference point for filtering. ISO format.

    Returns:
    --------
    tuple
        (DataFrame with 70-day processed data, DataFrame with 14-day processed data)
    """
    raw_records_file = os.path.join(data_dir, "Records.json")
    preprocessed_coords = preprocess_records_json(raw_records_file)

    # Filter coordinates for 70 days
    filtered_coords_70d = filter_coords(
        coords=preprocessed_coords,
        center_date=svy_date,
        start_adjust_days=-42,
        duration_days=70,
        accuracy_m=200,
        in_hk=True,
        to_gmt8=True,
        geom_hk=geom_hk,
    )

    # Convert to DataFrame
    df_70d = pd.DataFrame(filtered_coords_70d)
    df_70d["timestamp"] = pd.to_datetime(df_70d["timestamp"])
    df_70d = df_70d.set_index("timestamp")

    df_filtered_70d = filter_by_speed(df_70d)
    df_downsampled_70d = downsample_coordinates(df_filtered_70d)
    df_imputed_70d = impute_missing_data(df_downsampled_70d)
    df_stationary_70d = identify_stationary_points(df_imputed_70d)
    df_clustered_70d = cluster_stationary_points(df_stationary_70d)
    df_final_70d = add_home_work_clusters(df_clustered_70d)

    # Ensure the index is a DatetimeIndex with UTC timezone
    df_final_70d.index = pd.to_datetime(df_final_70d.index).tz_localize("UTC")

    # Export 70-day data to CSV
    df_final_70d.to_csv("./output/processed_records_70d.csv", index=False)

    # Filter for 14 days before survey date using the timestamp column
    # Ensure the timestamp column is in datetime format and timezone-naive
    df_final_70d["timestamp"] = pd.to_datetime(
        df_final_70d["timestamp"]
    ).dt.tz_localize(None)

    # Ensure start_14d and end_14d are timezone-naive
    start_14d = pd.to_datetime(svy_date).tz_localize(None) - datetime.timedelta(days=14)
    end_14d = pd.to_datetime(svy_date).tz_localize(None)

    # Filter rows based on the timestamp column
    df_14d = df_final_70d[
        (df_final_70d["timestamp"] >= start_14d)
        & (df_final_70d["timestamp"] <= end_14d)
    ]

    # Export 14-day data to CSV
    df_14d.to_csv("./output/processed_records_14d.csv", index=False)

    return df_final_70d, df_14d


### Process semantics
def process_semantics(data_dir, svy_date):
    """
    Process semantic data from JSON files in the Semantic Location History folder.

    Parameters:
    data_dir (str): Path to the Semantic Location History folder.
    svy_date (str): Survey date in ISO format

    Returns:
    pd.DataFrame: DataFrame containing processed place visit data for the 14-day period
        before the survey date, with columns including latitude, longitude, placeId,
        address, name, confidence metrics, timestamps, duration in seconds, and
        semantic type.


    """
    # Convert svy_date to datetime object
    if isinstance(svy_date, str):
        svy_date = datetime.datetime.strptime(svy_date, "%Y-%m-%dT%H:%M:%SZ")

    def process_timestamps(visit):
        """Process and standardize timestamps."""
        duration = visit["duration"]
        start_time = duration.get("startTimestamp") or duration.get("startTimestampMs")
        end_time = duration.get("endTimestamp") or duration.get("endTimestampMs")

        def parse_timestamp(timestamp):
            """Parse a timestamp string with or without fractional seconds."""
            if isinstance(timestamp, str):
                try:
                    # Try parsing with fractional seconds
                    return datetime.datetime.strptime(
                        timestamp.rstrip("Z"), "%Y-%m-%dT%H:%M:%S.%f"
                    )
                except ValueError:
                    # Fallback to parsing without fractional seconds
                    return datetime.datetime.strptime(
                        timestamp.rstrip("Z"), "%Y-%m-%dT%H:%M:%S"
                    )
            else:
                # Handle timestamps in milliseconds
                return datetime.datetime.fromtimestamp(int(timestamp) / 1000.0)

        # Parse start and end times
        start_dt = parse_timestamp(start_time)
        end_dt = parse_timestamp(end_time)

        # Convert to GMT+8
        start_dt = start_dt + datetime.timedelta(hours=8)
        end_dt = end_dt + datetime.timedelta(hours=8)

        return start_dt, end_dt

    # Find all JSON files in the Semantic Location History folder
    json_paths = []
    for root, dirs, files in os.walk(data_dir):
        json_paths.extend([os.path.join(root, f) for f in files if f.endswith(".json")])

    # Process place visits
    place_visits = []
    for path in json_paths:
        with open(path, "r") as f:
            data = json.load(f).get("timelineObjects", [])
            for obj in data:
                if "placeVisit" not in obj:
                    continue

                visit = obj["placeVisit"]
                if "location" not in visit or "duration" not in visit:
                    continue

                # Determine confidence label and accuracy list
                confidence_label = "placeConfidence"
                accuracy_list = ["MEDIUM_CONFIDENCE", "HIGH_CONFIDENCE"]

                # Process timestamps
                start_dt, end_dt = process_timestamps(visit)

                # Filter for 14-day window before survey date
                if not (svy_date - datetime.timedelta(days=14) <= start_dt <= svy_date):
                    continue

                # Process coordinates
                location = visit["location"]
                if "latitudeE7" in location:
                    location["latitude"] /= 1e7
                if "longitudeE7" in location:
                    location["longitude"] /= 1e7

                # Filter by location within Hong Kong
                point = Point(location["longitude"], location["latitude"])
                if not geom_hk.intersects(point):
                    continue

                # Filter by confidence level
                if visit.get(confidence_label) not in accuracy_list:
                    continue

                # Add processed visit to the list
                place_visits.append(
                    {
                        "latitude": location.get("latitude"),
                        "longitude": location.get("longitude"),
                        "placeId": location.get("placeId"),
                        "address": location.get("address"),
                        "name": location.get("name"),
                        "locationConfidence": location.get("locationConfidence"),
                        "placeConfidence": visit.get("placeConfidence"),
                        "startTimestampStandardised": start_dt,
                        "endTimestampStandardised": end_dt,
                        "seconds": (end_dt - start_dt).total_seconds(),
                        "semanticType": location.get("semanticType"),
                    }
                )

    # Create DataFrame and save to CSV
    df = pd.DataFrame(place_visits)
    df.to_csv("./output/processed_semantics_14d.csv", index=False)
    return df


## Main
def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Process geolocation data")
    parser.add_argument(
        "--data-dir", required=True, help="Directory with geolocation data"
    )
    parser.add_argument("--svy-date", required=True, help="Survey date (ISO format)")
    parser.add_argument("--api-key", help="Google Places API key (optional)")
    parser.add_argument(
        "--skip-indices", action="store_true", help="Skip mobility indices calculation"
    )

    args = parser.parse_args()

    # Process records and semantics
    records_70d, records_14d = process_records(args.data_dir, args.svy_date)
    semantics_14d = process_semantics(args.data_dir, args.svy_date)

    # Calculate mobility indices
    if not args.skip_indices:
        indices = calculate_mobility_indices(
            "./output/processed_records_14d.csv",
            "./output/processed_records_70d.csv",
            "./output/processed_semantics_14d.csv",
            args.api_key,
        )

        # Save indices to file
        with open("./output/mobility_indices.json", "w") as f:
            json.dump(indices, f, indent=2)


if __name__ == "__main__":
    main()
