#!/usr/bin/env python3

import argparse
import json
import pandas as pd
import numpy as np
import statistics
import datetime
import requests
import time
from datetime import timedelta
from sklearn.cluster import KMeans


def simple_euclidean_distance_km(lat1, lon1, lat2, lon2):
    """Calculate Euclidean distance between two points in km."""
    if pd.isnull(lat1) or pd.isnull(lon1) or pd.isnull(lat2) or pd.isnull(lon2):
        return np.nan
    lat_diff_km = abs(lat1 - lat2) * 111
    lon_diff_km = abs(lon1 - lon2) * 102
    return np.sqrt(lat_diff_km**2 + lon_diff_km**2)


def cal_trip_n(preprocessed_14d_data):
    """Calculate number of trips in the data."""
    mask = preprocessed_14d_data["transitioning"]
    transitions = mask.diff().ne(0)
    groups = mask.groupby(transitions.cumsum())
    trips = sum((1 for _, group in groups if group.iloc[0] and len(group) >= 2))
    return trips


def calculate_median_trip_duration(preprocessed_14d_data):
    """Calculate the median duration of trips in minutes."""
    mask = preprocessed_14d_data["transitioning"]
    transitions = mask.diff().ne(0)
    trip_groups = mask.groupby(transitions.cumsum())
    trip_durations = [
        len(group) * 5 for _, group in trip_groups if group.iloc[0] and len(group) >= 2
    ]
    if not trip_durations:
        return None
    return np.median(trip_durations)


def calculate_median_trip_distance(preprocessed_14d_data):
    """Calculate the median distance of trips."""
    mask = preprocessed_14d_data["transitioning"]
    transitions = mask.diff().ne(0)
    group_labels = transitions.cumsum()
    transition_counts = mask.groupby(group_labels).sum()
    valid_trip_groups = transition_counts[transition_counts >= 2].index
    trip_distances = (
        preprocessed_14d_data.groupby(group_labels)["distance"]
        .sum()
        .loc[valid_trip_groups]
        .values
    )
    if len(trip_distances) == 0:
        return None
    return np.median(trip_distances)


def calculate_total_stationary_duration(preprocessed_14d_data):
    """Calculate total time spent in stationary states in hours."""
    data = preprocessed_14d_data.copy()
    data["stationary"] = data["stationary"].astype(bool)
    data["block_id"] = (data["stationary"].shift(1) != data["stationary"]).cumsum()
    stationary_blocks = (
        data[data["stationary"]]
        .groupby(["cluster_label", "block_id"])
        .agg(start_time=("timestamp", "first"), end_time=("timestamp", "last"))
    )
    if stationary_blocks.empty:
        return 0
    total_duration = (
        stationary_blocks["end_time"] - stationary_blocks["start_time"]
    ).dt.total_seconds().sum() / 3600
    return total_duration


def calculate_median_outing_duration(data_70d, data_14d, max_outing_hours=24):
    """Calculate median duration of outings (periods between home stays)."""

    def identify_home_cluster(data_70d):
        valid_home_clusters = data_70d["assumed_home_cluster"].dropna()
        if len(valid_home_clusters) > 0:
            return valid_home_clusters.mode()[0]
        else:
            return None

    def calculate_outing_durations(data_14d, home_cluster, max_hours):
        data_14d["stationary"] = data_14d["stationary"].astype(bool)
        data_14d.loc[data_14d["stationary"], "is_home"] = (
            data_14d.loc[data_14d["stationary"], "cluster_label"] == home_cluster
        )
        data_14d.loc[data_14d["stationary"], "trip_group"] = (
            data_14d.loc[data_14d["stationary"], "is_home"].shift()
            != data_14d.loc[data_14d["stationary"], "is_home"]
        ).cumsum()
        home_ends = (
            data_14d.loc[data_14d["stationary"] & data_14d["is_home"]]
            .groupby("trip_group")["timestamp"]
            .last()
        )
        next_home_starts = (
            data_14d.loc[data_14d["stationary"] & data_14d["is_home"]]
            .groupby("trip_group")["timestamp"]
            .first()
            .shift(-1)
        )
        outing_durations = next_home_starts - home_ends
        outing_hours = outing_durations.dt.total_seconds() / 3600
        valid_outings = outing_hours[outing_hours <= max_hours]
        return valid_outings.tolist()

    home_cluster = identify_home_cluster(data_70d)
    outing_durations = calculate_outing_durations(
        data_14d, home_cluster, max_outing_hours
    )
    if not outing_durations:
        return None
    return statistics.median(outing_durations)


def calculate_median_outing_places(data_70d, data_14d, max_outing_hours=24):
    """Calculate median number of unique places visited during outings."""

    def identify_home_cluster(data_70d):
        valid_home_clusters = data_70d["assumed_home_cluster"].dropna()
        if len(valid_home_clusters) > 0:
            return valid_home_clusters.mode()[0]
        else:
            return None

    def calculate_places_per_outing(data, home_cluster, max_hours):
        data["stationary"] = data["stationary"].astype(bool)

        stationary_records = data.loc[data["stationary"]]

        data.loc[data["stationary"], "is_home"] = (
            stationary_records["cluster_label"] == home_cluster
        )
        data.loc[data["stationary"], "trip_group"] = (
            stationary_records["is_home"].shift() != stationary_records["is_home"]
        ).cumsum()

        # Calculate home_ends and next_home_starts
        home_ends = (
            stationary_records[stationary_records["is_home"]]
            .groupby("trip_group")["timestamp"]
            .last()
        )
        next_home_starts = (
            stationary_records[stationary_records["is_home"]]
            .groupby("trip_group")["timestamp"]
            .first()
            .shift(-1)
        )

        # Calculate outing durations
        outing_durations = next_home_starts - home_ends
        outing_hours = outing_durations.dt.total_seconds() / 3600
        valid_outing_groups = outing_hours[outing_hours <= max_hours].index + 1

        # Calculate places per outing
        places_per_outing = []
        for group in valid_outing_groups:
            outing_records = stationary_records[
                (stationary_records["trip_group"] == group)
                & (~stationary_records["is_home"])
            ]
            unique_places = len(outing_records["cluster_label"].unique())
            places_per_outing.append(unique_places)

        return places_per_outing

    home_cluster = identify_home_cluster(data_70d)
    places_per_outing = calculate_places_per_outing(
        data_14d, home_cluster, max_outing_hours
    )
    if not places_per_outing:
        return None
    return statistics.median(places_per_outing)


def calculate_walking_transition_percentage(preprocessed_14d_data):
    """Calculate the percentage of transitioning intervals that are likely walking."""
    valid_transitions = preprocessed_14d_data[
        ~preprocessed_14d_data["speed_km_h"].isin([-1, "", np.nan])
        & preprocessed_14d_data["transitioning"]
    ]
    if valid_transitions.empty:
        return None
    walking_percentage = (valid_transitions["speed_km_h"] <= 5.58).mean()
    return walking_percentage


def calculate_rush_hour_percentage(data_70d, data_14d):
    """Calculate percentage of weekday outings that start during rush hours."""

    def identify_home_cluster(data_70d):
        valid_home_clusters = data_70d["assumed_home_cluster"].dropna()
        if len(valid_home_clusters) > 0:
            return valid_home_clusters.mode()[0]
        else:
            return None

    def is_rush_hour(timestamp):
        time = timestamp.time()
        morning_rush = datetime.time(7, 30) <= time <= datetime.time(9, 30)
        lunch_rush = datetime.time(12, 30) <= time <= datetime.time(14, 0)
        evening_rush = datetime.time(17, 30) <= time <= datetime.time(19, 30)
        return morning_rush or lunch_rush or evening_rush

    def get_rush_hour_outings(data, home_cluster):
        data["weekday"] = data["timestamp"].dt.dayofweek < 5
        data["home_cluster"] = data["cluster_label"] == home_cluster
        data["trip_group"] = (
            data["home_cluster"].shift() != data["home_cluster"]
        ).cumsum()
        non_home_starts = (
            data[~data["home_cluster"] & data["weekday"]]
            .groupby("trip_group")["timestamp"]
            .first()
        )
        rush_hour_flags = [is_rush_hour(start_time) for start_time in non_home_starts]
        return rush_hour_flags

    if data_14d is None or data_14d.empty:
        return None

    home_cluster = identify_home_cluster(data_70d)
    if home_cluster is None or home_cluster == -1:
        return None

    rush_hour_flags = get_rush_hour_outings(data_14d, home_cluster)
    if not rush_hour_flags:
        return None

    return (sum(rush_hour_flags) / len(rush_hour_flags)) * 100


def calculate_home_leave_return_time(data_70d, data_14d):
    """Calculate median home arrival time."""

    def identify_home_cluster(data_70d):
        valid_home_clusters = data_70d["assumed_home_cluster"].dropna()
        if len(valid_home_clusters) > 0:
            return valid_home_clusters.mode()[0]
        else:
            return None

    def calculate_return_times(data, home_cluster):
        if home_cluster is None or home_cluster in [np.nan, -1, ""]:
            return []
        data = data[
            ~data["cluster_label"].isin([-1, "", np.nan])
            & data["cluster_label"].notna()
        ]
        data = data.assign(home_cluster=lambda x: x["cluster_label"] == home_cluster)
        data.loc[:, "trip_grouping"] = (
            data["home_cluster"].shift() != data["home_cluster"]
        ).cumsum()
        home_blocks = data[data["home_cluster"]]
        start_times = home_blocks.groupby("trip_grouping")["timestamp"].first()
        return_hours = [(t.hour - 3 + 24) % 24 + t.minute / 60.0 for t in start_times]
        return return_hours

    home_cluster = identify_home_cluster(data_70d)
    return_hours = calculate_return_times(data_14d, home_cluster)
    if return_hours:
        median_return_time = statistics.median(return_hours)
    else:
        median_return_time = None
    return median_return_time


def calculate_home_leave_time(data_70d, data_14d):
    """Calculate median home departure time."""

    def identify_home_cluster(data_70d):
        valid_home_clusters = data_70d["assumed_home_cluster"].dropna()
        if len(valid_home_clusters) > 0:
            return valid_home_clusters.mode()[0]
        else:
            return None

    def calculate_departure_times(data, home_cluster):
        if home_cluster is None or home_cluster in [np.nan, -1, ""]:
            return []
        data = data[
            ~data["cluster_label"].isin([-1, "", np.nan])
            & data["cluster_label"].notna()
        ]
        data = data.assign(home_cluster=lambda x: x["cluster_label"] == home_cluster)
        data.loc[:, "trip_grouping"] = (
            data["home_cluster"].shift() != data["home_cluster"]
        ).cumsum()
        home_blocks = data[data["home_cluster"]]
        end_times = home_blocks.groupby("trip_grouping")["timestamp"].last()
        leave_hours = [(t.hour - 3 + 24) % 24 + t.minute / 60.0 for t in end_times]
        return leave_hours

    home_cluster = identify_home_cluster(data_70d)
    leave_hours = calculate_departure_times(data_14d, home_cluster)
    if leave_hours:
        median_leave_time = statistics.median(leave_hours)
    else:
        median_leave_time = None
    return median_leave_time


def calculate_total_home_time(data_70d, data_14d):
    """Calculate total time spent at home location during 14-day period."""

    def identify_home_cluster(data_70d):
        valid_home_clusters = data_70d["assumed_home_cluster"].dropna()
        if len(valid_home_clusters) > 0:
            return valid_home_clusters.mode()[0]
        else:
            return None

    def calculate_home_durations(data, home_cluster):
        data["stationary"] = data["stationary"].astype(bool)
        data["block_id"] = (data["stationary"].shift(1) != data["stationary"]).cumsum()
        stationary_blocks = data[data["stationary"]]
        durations = (
            stationary_blocks.groupby(["cluster_label", "block_id"])
            .agg(
                timestamp_first=("timestamp", "first"),
                timestamp_last=("timestamp", "last"),
            )
            .reset_index()
        )
        durations["duration"] = (
            durations["timestamp_last"] - durations["timestamp_first"]
        ).dt.total_seconds() / 60
        home_durations = durations[durations["cluster_label"] == home_cluster][
            "duration"
        ]
        home_durations = home_durations[home_durations.notna()]
        return home_durations.tolist()

    home_cluster = identify_home_cluster(data_70d)
    home_durations = calculate_home_durations(data_14d, home_cluster)
    total_home_time = sum(home_durations) if home_durations else 0
    return total_home_time


def calculate_median_nonhome_stationary_distance(data_70d, data_14d):
    """Calculate median distance from home for non-home stationary periods."""

    def identify_home_cluster(data_70d):
        valid_home_clusters = data_70d["assumed_home_cluster"].dropna()
        if len(valid_home_clusters) > 0:
            return valid_home_clusters.mode()[0]
        else:
            return None

    def calculate_cluster_distances(data, home_cluster):
        cluster_centroids = (
            data.groupby("cluster_label")
            .agg({"latitude": "mean", "longitude": "mean"})
            .reset_index()
        )
        home_coords = cluster_centroids[
            cluster_centroids["cluster_label"] == home_cluster
        ]
        if home_coords.empty:
            return pd.DataFrame(columns=["cluster_label", "distance_from_home"])

        def get_distance_from_home(row):
            if row["cluster_label"] == home_cluster:
                return 0
            return simple_euclidean_distance_km(
                row["latitude"],
                row["longitude"],
                home_coords["latitude"].iloc[0],
                home_coords["longitude"].iloc[0],
            )

        cluster_centroids["distance_from_home"] = cluster_centroids.apply(
            get_distance_from_home, axis=1
        )
        return cluster_centroids[["cluster_label", "distance_from_home"]]

    def get_stay_distances(data, cluster_distances):
        data["stationary"] = data["stationary"].astype(bool)
        data["block_id"] = (data["stationary"].shift(1) != data["stationary"]).cumsum()
        stationary_blocks = data[data["stationary"]]
        stay_periods = (
            stationary_blocks.groupby(["cluster_label", "block_id"])
            .agg(
                timestamp_first=("timestamp", "first"),
                timestamp_last=("timestamp", "last"),
            )
            .reset_index()
        )
        stay_periods["duration"] = (
            stay_periods["timestamp_last"] - stay_periods["timestamp_first"]
        )
        stay_periods = stay_periods.merge(
            cluster_distances[["cluster_label", "distance_from_home"]],
            on="cluster_label",
            how="left",
        )
        nonhome_distances = [d for d in stay_periods["distance_from_home"] if d > 0]
        return nonhome_distances

    home_cluster = identify_home_cluster(data_70d)
    cluster_distances = calculate_cluster_distances(data_70d, home_cluster)
    nonhome_distances = get_stay_distances(data_14d, cluster_distances)
    if not nonhome_distances:
        return None
    return statistics.median(nonhome_distances)


def calculate_total_work_time(data_70d, data_14d):
    """Calculate total time spent at work location during 14-day period."""

    def identify_work_cluster(data_70d):
        valid_work_clusters = data_70d["assumed_work_cluster"].dropna()
        if len(valid_work_clusters) > 0:
            return valid_work_clusters.mode()[0]
        else:
            return None

    def calculate_work_durations(data, work_cluster):
        data["stationary"] = data["stationary"].astype(bool)
        data["block_id"] = (data["stationary"].shift(1) != data["stationary"]).cumsum()
        stationary_blocks = data[data["stationary"]]
        durations = (
            stationary_blocks.groupby(["cluster_label", "block_id"])
            .agg(
                timestamp_first=("timestamp", "first"),
                timestamp_last=("timestamp", "last"),
            )
            .reset_index()
        )
        durations["duration"] = (
            durations["timestamp_last"] - durations["timestamp_first"]
        ).dt.total_seconds() / 60
        work_durations = durations[durations["cluster_label"] == work_cluster][
            "duration"
        ]
        work_durations = work_durations[work_durations.notna()]
        return work_durations.tolist()

    work_cluster = identify_work_cluster(data_70d)
    work_durations = calculate_work_durations(data_14d, work_cluster)
    total_work_time = sum(work_durations) if work_durations else 0
    return total_work_time


def calculate_unique_places(data_14d):
    """Calculate the number of unique places (clusters) visited during the 14-day period."""

    def get_unique_clusters(data):
        data["stationary"] = data["stationary"].astype(bool)
        data["block_id"] = (data["stationary"].shift(1) != data["stationary"]).cumsum()
        stationary_blocks = data[data["stationary"]]
        clusters = (
            stationary_blocks.groupby(["cluster_label", "block_id"])
            .first()
            .reset_index()["cluster_label"]
            .dropna()
            .tolist()
        )
        return clusters

    clusters = get_unique_clusters(data_14d)
    valid_clusters = [c for c in clusters if c not in [-1, "", np.nan]]
    unique_count = len(set(valid_clusters))
    return unique_count


# Place type categories for indices 15-20
place_type_categories = {
    "Recreational_visits": [
        "art_gallery",
        "museum",
        "performing_arts_theater",
        "amusement_center",
        "amusement_park",
        "aquarium",
        "banquet_hall",
        "bowling_alley",
        "casino",
        "community_center",
        "convention_center",
        "cultural_center",
        "dog_park",
        "event_venue",
        "hiking_area",
        "historical_landmark",
        "marina",
        "movie_rental",
        "movie_theater",
        "national_park",
        "night_club",
        "park",
        "tourist_attraction",
        "visitor_center",
        "wedding_venue",
        "zoo",
        "athletic_field",
        "fitness_center",
        "golf_course",
        "gym",
        "playground",
        "ski_resort",
        "sports_club",
        "sports_complex",
        "stadium",
        "swimming_pool",
        "american_restaurant",
        "bakery",
        "bar",
        "barbecue_restaurant",
        "brazilian_restaurant",
        "breakfast_restaurant",
        "brunch_restaurant",
        "cafe",
        "chinese_restaurant",
        "coffee_shop",
        "fast_food_restaurant",
        "french_restaurant",
        "greek_restaurant",
        "hamburger_restaurant",
        "ice_cream_shop",
        "indian_restaurant",
        "indonesian_restaurant",
        "italian_restaurant",
        "japanese_restaurant",
        "korean_restaurant",
        "lebanese_restaurant",
        "meal_delivery",
        "meal_takeaway",
        "mediterranean_restaurant",
        "mexican_restaurant",
        "middle_eastern_restaurant",
        "pizza_restaurant",
        "ramen_restaurant",
        "restaurant",
        "sandwich_shop",
        "seafood_restaurant",
        "spanish_restaurant",
        "steak_house",
        "sushi_restaurant",
        "thai_restaurant",
        "turkish_restaurant",
        "vegan_restaurant",
        "vegetarian_restaurant",
        "vietnamese_restaurant",
    ],
    "Errand_visits": [
        "accounting",
        "atm",
        "bank",
        "barber_shop",
        "beauty_salon",
        "cemetery",
        "child_care_agency",
        "consultant",
        "courier_service",
        "electrician",
        "florist",
        "funeral_home",
        "hair_care",
        "hair_salon",
        "insurance_agency",
        "laundry",
        "lawyer",
        "locksmith",
        "moving_company",
        "painter",
        "plumber",
        "real_estate_agency",
        "roofing_contractor",
        "storage",
        "tailor",
        "telecommunications_service_provider",
        "travel_agency",
        "veterinary_care",
    ],
    "Transport_hub_visits": [
        "airport",
        "bus_station",
        "bus_stop",
        "ferry_terminal",
        "heliport",
        "light_rail_station",
        "park_and_ride",
        "subway_station",
        "taxi_stand",
        "train_station",
        "transit_depot",
        "transit_station",
        "truck_stop",
    ],
    "Education_visits": [
        "library",
        "preschool",
        "primary_school",
        "school",
        "secondary_school",
        "university",
    ],
    "Health_visits": [
        "dental_clinic",
        "dentist",
        "doctor",
        "drugstore",
        "hospital",
        "medical_lab",
        "pharmacy",
        "physiotherapist",
        "spa",
    ],
    "Shop_visits": [
        "auto_parts_store",
        "bicycle_store",
        "book_store",
        "cell_phone_store",
        "clothing_store",
        "convenience_store",
        "department_store",
        "discount_store",
        "electronics_store",
        "furniture_store",
        "gift_shop",
        "grocery_store",
        "hardware_store",
        "home_goods_store",
        "home_improvement_store",
        "jewelry_store",
        "liquor_store",
        "market",
        "pet_store",
        "shoe_store",
        "shopping_mall",
        "sporting_goods_store",
        "store",
        "supermarket",
        "wholesaler",
    ],
}


def get_semantic_place_types(df, api_key, place_type_category_dict):
    """Get place types from Google Places API and categorize visits."""
    if not api_key:
        return {
            "recreational_visits": None,
            "errand_visits": None,
            "transport_hub_visits": None,
            "education_visits": None,
            "health_visits": None,
            "shop_visits": None,
        }

    # Get unique place IDs
    place_ids = df["placeId"].dropna().unique()
    place_types = {}

    # Call API for each place ID
    for place_id in place_ids:
        try:
            response = requests.get(
                "https://maps.googleapis.com/maps/api/place/details/json",
                params={"place_id": place_id, "fields": "types", "key": api_key},
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "OK":
                    place_types[place_id] = result.get("result", {}).get("types", [])

            time.sleep(0.1)

        except Exception as e:
            print(f"Error getting place types for {place_id}: {str(e)}")
            continue

    # Add place types to DataFrame
    df["place_type"] = df["placeId"].map(place_types)

    # Initialize category columns
    category_mapping = {
        "Recreational_visits": "recreational_visits",
        "Errand_visits": "errand_visits",
        "Transport_hub_visits": "transport_hub_visits",
        "Education_visits": "education_visits",
        "Health_visits": "health_visits",
        "Shop_visits": "shop_visits",
    }

    for category in category_mapping.keys():
        df[category] = False

    # Categorize places
    for idx, row in df.iterrows():
        if pd.isna(row["place_type"]) or not row["place_type"]:
            continue

        place_type_list = row["place_type"]
        for category, types in place_type_category_dict.items():
            if any(t in types for t in place_type_list):
                df.loc[idx, category] = True

    # Count visits for each category
    category_counts = {
        category_mapping[category]: int(df[category].sum())
        for category in category_mapping.keys()
    }

    return category_counts


def calculate_mobility_indices(
    records_14d_path, records_70d_path, semantics_14d_path=None, api_key=None
):
    """Calculate mobility indices from preprocessed geolocation data."""

    records_14d = pd.read_csv(records_14d_path, parse_dates=["timestamp"])
    records_70d = pd.read_csv(records_70d_path, parse_dates=["timestamp"])

    # Load semantics data if provided
    semantics_14d = None
    if semantics_14d_path:
        try:
            semantics_14d = pd.read_csv(semantics_14d_path)
        except Exception as e:
            print(f"Warning: Could not load semantics data: {str(e)}")

    # Dict to store results
    mobility_indices = {}

    # 1. Trip count
    mobility_indices["trip_count"] = cal_trip_n(records_14d)

    # 2. Median trip duration
    mobility_indices["median_trip_duration"] = calculate_median_trip_duration(
        records_14d
    )

    # 3. Median trip distance
    mobility_indices["median_trip_distance"] = calculate_median_trip_distance(
        records_14d
    )

    # 4. Total stationary time
    mobility_indices["total_stationary_time"] = calculate_total_stationary_duration(
        records_14d
    )

    # 5. Median outing duration
    mobility_indices["median_outing_duration"] = calculate_median_outing_duration(
        records_70d, records_14d
    )

    # 6. Median outing places
    mobility_indices["median_outing_places"] = calculate_median_outing_places(
        records_70d, records_14d
    )

    # 7. On-foot percentage
    mobility_indices["on_foot_percentage"] = calculate_walking_transition_percentage(
        records_14d
    )

    # 8. Rush hour outings percent
    mobility_indices["rush_hour_outings_percent"] = calculate_rush_hour_percentage(
        records_70d, records_14d
    )

    # 9. Median home arrival time
    mobility_indices["median_home_arrival_time"] = calculate_home_leave_return_time(
        records_70d, records_14d
    )

    # 10. Median home departure time
    mobility_indices["median_home_departure_time"] = calculate_home_leave_time(
        records_70d, records_14d
    )

    # 11. Total home time
    mobility_indices["total_home_time"] = calculate_total_home_time(
        records_70d, records_14d
    )

    # 12. Median non-home distance
    mobility_indices["median_non_home_distance"] = (
        calculate_median_nonhome_stationary_distance(records_70d, records_14d)
    )

    # 13. Total work time
    mobility_indices["total_work_time"] = calculate_total_work_time(
        records_70d, records_14d
    )

    # 14. Unique places visited
    mobility_indices["unique_places_visited"] = calculate_unique_places(records_14d)

    # 15-20. Place type visit counts (if API key provided)
    if semantics_14d is not None:
        place_counts = get_semantic_place_types(
            semantics_14d, api_key, place_type_categories
        )
        mobility_indices.update(place_counts)
    else:
        # Set empty values if no semantics data
        mobility_indices.update(
            {
                "recreational_visits": None,
                "errand_visits": None,
                "transport_hub_visits": None,
                "education_visits": None,
                "health_visits": None,
                "shop_visits": None,
            }
        )

    return mobility_indices


def main():
    """Main function to parse arguments and execute processing."""
    parser = argparse.ArgumentParser(
        description="Calculate mobility indices from geolocation data"
    )
    parser.add_argument(
        "--records-14d", required=True, help="Path to 14-day records CSV file"
    )
    parser.add_argument(
        "--records-70d", required=True, help="Path to 70-day records CSV file"
    )
    parser.add_argument("--semantics-14d", help="Path to 14-day semantics file")
    parser.add_argument("--api-key", help="Google Places API key (optional)")

    args = parser.parse_args()

    # Calculate mobility indices
    indices = calculate_mobility_indices(
        args.records_14d, args.records_70d, args.semantics_14d, args.api_key
    )

    # Print results
    print("\nMobility Indices:")
    print(json.dumps(indices, indent=2))


if __name__ == "__main__":
    main()
