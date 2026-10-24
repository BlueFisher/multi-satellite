import dataclasses

import astropy.units as u


@dataclasses.dataclass
class SatelliteConfig:
    LOWEST_SATELLITE_RADIUS = 6578 * u.km
    HIGHEST_SATELLITE_RADIUS = 8378 * u.km
    RADIUS_DISTANCE = HIGHEST_SATELLITE_RADIUS - LOWEST_SATELLITE_RADIUS
