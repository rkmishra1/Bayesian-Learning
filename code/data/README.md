# Application data

`uk_links_2015_2019.csv.gz` is a panel of 17,139 motorway and A-road links in Great Britain
observed from 2015 to 2019 (85,695 link-years). It is built by `scripts/build_uk_data.py`
from three Department for Transport sources:

| Source | File |
|---|---|
| STATS19 road safety data, collisions 1979 to latest year | `dft-road-casualty-statistics-collision-1979-latest-published-year.csv` from data.dft.gov.uk |
| Road traffic statistics, annual average daily flow by count point | `dft_traffic_counts_aadf.zip` from roadtraffic.dft.gov.uk/downloads |
| Major Roads Database link geometry, 2019 | `mrdb-2019.zip` from roadtraffic.dft.gov.uk/downloads |

Contains public sector information licensed under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Construction

* A link is a count-point link of the 2019 Major Roads Database, a stretch of major road
  between two junctions with other major roads. Links are kept if the AADF series has all
  five years.
* Each injury collision whose first road is a motorway, A(M) or A road is assigned to the
  nearest link carrying the same road number, if it lies within 300 m of that link's
  straight-line chord. Of 312,386 such collisions in 2015 to 2019, 253,358 (81%) were
  assigned. The rest lie on links without a full AADF series, or further from the chord.
* `y` counts all injury collisions, `y_ksi` those with a fatal or serious casualty.
* `chord_km` is the straight-line length of the link in the database, `node_degree` the
  number of other major-road links meeting the link at its two ends, and `road_density` the
  length of major road within 2 km of the link midpoint (km per km^2).

## Columns

`cp`, `year`, `estimation_method` (AADF counted or estimated), traffic by vehicle class
(`all_motor_vehicles`, `all_HGVs`, `LGVs`, `buses_and_coaches`, `two_wheeled_motor_vehicles`,
`pedal_cycles`), `road_name`, `road_category` (TM/PM motorway, TA trunk A road, PA principal
A road), `region_name`, `local_authority_name`, `link_length_km`, `chord_km`, `node_degree`,
`road_density`, `y`, `y_ksi`.
