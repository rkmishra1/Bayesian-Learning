"""Build the British major-road panel used in the application (Section 6).

Sources (Open Government Licence v3.0, Department for Transport):
  * STATS19 collisions, 1979 to latest year:
    https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-1979-latest-published-year.csv
  * Annual average daily flow (AADF) by count point:
    https://storage.googleapis.com/dft-statistics/road-traffic/downloads/data-gov-uk/dft_traffic_counts_aadf.zip
  * Major Roads Database link geometry, 2019 (one straight line per count-point link):
    https://storage.googleapis.com/dft-statistics/road-traffic/mrdb-2019.zip

A site is a count-point link of the 2019 Major Roads Database: a stretch of motorway or
A road between two junctions with other major roads. Each collision on a motorway, A(M) or
A road is assigned to the nearest link carrying the same road number.

    python scripts/build_uk_data.py --collisions RAW/collisions.csv --aadf RAW/aadf.csv \
        --mrdb RAW/MRDB_2019_published.shp --out data/uk_links_2015_2019.csv.gz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

YEARS = range(2015, 2020)  # five pre-pandemic years
SNAP_MAX = 300.0  # metres: largest accepted distance from a collision to its link's chord
RADIUS = 2000.0  # metres: neighbourhood for the major-road density


def road_name(cls: pd.Series, num: pd.Series) -> pd.Series:
    num = num.astype(int).astype(str)
    out = pd.Series(pd.NA, index=cls.index, dtype="object")
    out[cls == 1] = "M" + num[cls == 1]
    out[cls == 2] = "A" + num[cls == 2] + "(M)"
    out[cls == 3] = "A" + num[cls == 3]
    return out


def load_collisions(path: Path) -> pd.DataFrame:
    cols = ["collision_index", "collision_year", "location_easting_osgr", "location_northing_osgr",
            "collision_severity", "first_road_class", "first_road_number"]
    parts = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=1_000_000, low_memory=False):
        ch = ch[ch.collision_year.isin(YEARS) & ch.first_road_class.isin([1, 2, 3])]
        ch = ch[(ch.first_road_number > 0) & ch.location_easting_osgr.notna() & ch.location_northing_osgr.notna()]
        parts.append(ch)
    c = pd.concat(parts, ignore_index=True)
    c["road"] = road_name(c.first_road_class, c.first_road_number)
    return c


def snap(c: pd.DataFrame, links: gpd.GeoDataFrame) -> pd.DataFrame:
    """Nearest link on the same road, by road number."""
    cp = np.full(len(c), -1, dtype=np.int64)
    dist = np.full(len(c), np.inf)
    pts = shapely.points(c.location_easting_osgr.values, c.location_northing_osgr.values)
    for road, idx in c.groupby("road").indices.items():
        lk = links[links.RoadNumber == road]
        if lk.empty:
            continue
        tree = shapely.STRtree(lk.geometry.values)
        (ip, il), d = tree.query_nearest(pts[idx], return_distance=True, all_matches=False)
        cp[idx[ip]] = lk.CP_Number.values[il]
        dist[idx[ip]] = d
    c = c.assign(cp=cp, snap_m=dist)
    return c


def link_features(links: gpd.GeoDataFrame) -> pd.DataFrame:
    geom = links.geometry.values
    chord = shapely.length(geom)
    # junction degree: other links sharing an end node (nodes matched to the nearest metre)
    ends = np.vstack([shapely.get_coordinates(shapely.get_point(geom, 0)),
                      shapely.get_coordinates(shapely.get_point(geom, -1))]).round(0)
    keys = pd.Series([f"{x:.0f}_{y:.0f}" for x, y in ends])
    deg = keys.map(keys.value_counts()).values - 1
    n = len(links)
    node_degree = deg[:n] + deg[n:]
    # major-road length within RADIUS of the link midpoint, km per km^2
    mid = shapely.line_interpolate_point(geom, 0.5, normalized=True)
    tree = shapely.STRtree(geom)
    buf = shapely.buffer(mid, RADIUS)
    pairs = tree.query(buf, predicate="intersects")
    clipped = shapely.length(shapely.intersection(geom[pairs[1]], buf[pairs[0]]))
    dens = np.bincount(pairs[0], weights=clipped, minlength=n) / 1000.0 / (np.pi * (RADIUS / 1000.0) ** 2)
    return pd.DataFrame({"cp": links.CP_Number.values.astype(np.int64), "chord_km": chord / 1000.0,
                         "node_degree": node_degree, "road_density": dens})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collisions", required=True)
    ap.add_argument("--aadf", required=True)
    ap.add_argument("--mrdb", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    links = gpd.read_file(a.mrdb)
    links["CP_Number"] = links.CP_Number.astype(np.int64)
    feats = link_features(links)

    aadf = pd.read_csv(a.aadf, low_memory=False)
    aadf = aadf[aadf.year.isin(YEARS) & (aadf.road_type == "Major") & aadf.count_point_id.isin(feats.cp)]
    full = aadf.groupby("count_point_id").year.nunique()
    aadf = aadf[aadf.count_point_id.isin(full.index[full == len(YEARS)])]
    # time-invariant attributes are taken from 2019, the year of the link geometry
    attr = aadf[aadf.year == 2019].set_index("count_point_id")[
        ["road_name", "road_category", "region_name", "local_authority_name", "link_length_km"]]

    col = load_collisions(Path(a.collisions))
    col = snap(col, links[links.CP_Number.isin(attr.index)])
    n_all = len(col)
    col = col[(col.cp >= 0) & (col.snap_m <= SNAP_MAX)]
    print(f"collisions on M/A(M)/A roads {min(YEARS)}-{max(YEARS)}: {n_all:,}; "
          f"assigned within {SNAP_MAX:.0f} m: {len(col):,} ({len(col) / n_all:.1%})")
    counts = col.groupby(["cp", "collision_year"]).agg(
        y=("collision_index", "size"), y_ksi=("collision_severity", lambda s: int((s <= 2).sum())))

    panel = aadf[["count_point_id", "year", "estimation_method", "all_motor_vehicles", "all_HGVs", "LGVs",
                  "buses_and_coaches", "two_wheeled_motor_vehicles", "pedal_cycles"]].rename(
        columns={"count_point_id": "cp"})
    panel = panel.merge(attr, left_on="cp", right_index=True).merge(feats, on="cp")
    panel = panel.merge(counts, left_on=["cp", "year"], right_index=True, how="left")
    panel[["y", "y_ksi"]] = panel[["y", "y_ksi"]].fillna(0).astype(int)
    panel = panel[panel.all_motor_vehicles > 0].sort_values(["cp", "year"])
    keep = panel.groupby("cp").year.transform("size") == len(YEARS)
    panel = panel[keep].reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(a.out, index=False, float_format="%.6g")
    print(f"links: {panel.cp.nunique():,}, link-years: {len(panel):,}, "
          f"collisions in panel: {panel.y.sum():,}, mean per link-year: {panel.y.mean():.2f}")


if __name__ == "__main__":
    main()
