import cdsapi
import os

client = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key=os.environ["CDSAPI_KEY"],  # REDACTED for release: personal CDS API key removed; set CDSAPI_KEY or use ~/.cdsapirc
)

dataset = "reanalysis-era5-pressure-levels"
output_dir = "/umbc/rs/pi_halem/users/yashd1/AURORA_DATA"
os.makedirs(output_dir, exist_ok=True)

common_request = {
    "product_type": ["reanalysis"],
    "variable": [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "year": ["2023"],
    "time": ["00:00", "06:00", "12:00", "18:00"],
    "pressure_level": [
        "50", "100", "150", "200", "250", "300",
        "400", "500", "600", "700", "850", "925", "1000"
    ],
    "data_format": "grib",
    "download_format": "unarchived"
}

# February
request_feb = common_request.copy()
request_feb["month"] = ["02"]
request_feb["day"] = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28"
]

client.retrieve(dataset, request_feb).download(
    os.path.join(output_dir, "era5_pressure_2023_02.grib")
)

# March
request_mar = common_request.copy()
request_mar["month"] = ["03"]
request_mar["day"] = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"
]

client.retrieve(dataset, request_mar).download(
    os.path.join(output_dir, "era5_pressure_2023_03.grib")
)
