#!/usr/bin/env python

"""
Store all coordinates in one place.

To chose a location for download, copy the location id into the relevant
line in `config.py`

In future think about:
 - keeping a record when we run a donwload with the date of download
   and a commit hash.
 - having an interface to the `config.py` file which allows the user
   to specify groups of coordinates to download in series/parallel.
   e.g. user specifies that they want to download all "spots"
   locations in Africa.

"""

import pandas as pd

# initialise a DataFrame to store coordinate
coordinate_store = pd.DataFrame(
    columns=["continent", "country", "type", "latitude", "longitude"]
)

# rename index
coordinate_store.index.names = ["id"]

# --------------------------------------------------------------------------------
# sudan
coordinate_store.loc["00"] = {
    "continent": "Africa",
    "country": "Sudan",
    "type": "spots",
    "latitude": 11.58,
    "longitude": 27.94,
}
coordinate_store.loc["01"] = {
    "continent": "Africa",
    "country": "Sudan",
    "type": "labyrinths",
    "latitude": 11.12,
    "longitude": 28.37,
}
coordinate_store.loc["02"] = {
    "continent": "Africa",
    "country": "Sudan",
    "type": "gaps",
    "latitude": 10.96,
    "longitude": 28.20,
}

# niger
coordinate_store.loc["03"] = {
    "continent": "Africa",
    "country": "Niger",
    "type": "labyrinths",
    "latitude": 13.12,
    "longitude": 2.59,
}
coordinate_store.loc["04"] = {
    "continent": "Africa",
    "country": "Niger",
    "type": "labyrinths",
    "latitude": 13.17,
    "longitude": 1.58,
}

# senegal
coordinate_store.loc["05"] = {
    "continent": "Africa",
    "country": "Senegal",
    "type": "labyrinths",
    "latitude": 15.20,
    "longitude": -15.20,
}
coordinate_store.loc["06"] = {
    "continent": "Africa",
    "country": "Senegal",
    "type": "labyrinths",
    "latitude": 15.09,
    "longitude": -15.04,
}
coordinate_store.loc["07"] = {
    "continent": "Africa",
    "country": "Senegal",
    "type": "gaps",
    "latitude": 15.80,
    "longitude": -14.36,
}
coordinate_store.loc["08"] = {
    "continent": "Africa",
    "country": "Senegal",
    "type": "gaps",
    "latitude": 15.11,
    "longitude": -14.53,
}

# zambia
coordinate_store.loc["09"] = {
    "continent": "Africa",
    "country": "Zambia",
    "type": "gaps",
    "latitude": -15.34,
    "longitude": 22.22,
}

# kenya
coordinate_store.loc["10"] = {
    "continent": "Africa",
    "country": "Kenya",
    "type": "spots",
    "latitude": 0.43,
    "longitude": 40.30,
}

# somalia
coordinate_store.loc["11"] = {
    "continent": "Africa",
    "country": "Somalia",
    "type": "labyrinths",
    "latitude": 8.09,
    "longitude": 47.44,
}

# australia
coordinate_store.loc["12"] = {
    "continent": "Australia",
    "country": "Australia",
    "type": "gaps",
    "latitude": -15.71,
    "longitude": 133.10,
}

# usa
coordinate_store.loc["13"] = {
    "continent": "America",
    "country": "Mexico",
    "type": "labyrinths",
    "latitude": 26.82,
    "longitude": -112.86,
}
coordinate_store.loc["14"] = {
    "continent": "America",
    "country": "USA",
    "type": "labyrinths",
    "latitude": 31.05,
    "longitude": -103.09,
}

# --------------------------------------------------------------------------------

# declining locations in West Niger
coordinate_store.loc["15"] = {
    "continent": "Africa",
    "country": "Niger",
    "type": "labyrinths",
    "latitude": 14.91,
    "longitude": -0.66,
}
coordinate_store.loc["16"] = {
    "continent": "Africa",
    "country": "Niger",
    "type": "spots",
    "latitude": 15.03,
    "longitude": -0.87,
}
coordinate_store.loc["17"] = {
    "continent": "Africa",
    "country": "Niger",
    "type": "declining",
    "latitude": 15.23,
    "longitude": -0.97,
}
coordinate_store.loc["18"] = {
    "continent": "Africa",
    "country": "Niger",
    "type": "spots/labyrinths",
    "latitude": 15.34,
    "longitude": -1.15,
}
coordinate_store.loc["19"] = {
    "continent": "Africa",
    "country": "Mali",
    "type": "declining",
    "latitude": 15.14,
    "longitude": -1.16,
}
coordinate_store.loc["20"] = {
    "continent": "Africa",
    "country": "Mali",
    "type": "spots/labyrinths",
    "latitude": 14.85,
    "longitude": -1.43,
}
coordinate_store.loc["21"] = {
    "continent": "Africa",
    "country": "Mali",
    "type": "spots/labyrinths",
    "latitude": 14.97,
    "longitude": -1.12,
}
coordinate_store.loc["22"] = {
    "continent": "Africa",
    "country": "Mali",
    "type": "labyrinths",
    "latitude": 15.14,
    "longitude": -1.56,
}
coordinate_store.loc["23"] = {
    "continent": "Africa",
    "country": "Niger",
    "type": "spots/labyrinths",
    "latitude": 15.02,
    "longitude": -1.35,
}

# degraded locations in West Niger
coordinate_store.loc["24"] = {
    "continent": "Africa",
    "country": "Mali",
    "type": "degraded",
    "latitude": 16.26,
    "longitude": -1.83,
}
coordinate_store.loc["25"] = {
    "continent": "Africa",
    "country": "Mali",
    "type": "spots/labyrinths",
    "latitude": 16.19,
    "longitude": -1.83,
}
coordinate_store.loc["26"] = {
    "continent": "Africa",
    "country": "Mali",
    "type": "spots/labyrinths",
    "latitude": 16.17,
    "longitude": -2.03,
}
coordinate_store.loc["27"] = {
    "continent": "Africa",
    "country": "Mali",
    "type": "spots/labyrinths",
    "latitude": 16.48,
    "longitude": -1.87,
}
coordinate_store.loc["28"] = {
    "continent": "Africa",
    "country": "Mali",
    "type": "spots/labyrinths",
    "latitude": 15.95,
    "longitude": -1.52,
}
coordinate_store.loc["29"] = {
    "continent": "Africa",
    "country": "Mali",
    "type": "spots/labyrinths",
    "latitude": 15.86,
    "longitude": -2.05,
}

# healthy locations in West Niger
coordinate_store.loc["30"] = {
    "continent": "Africa",
    "country": "Mali",
    "type": "labyrinths",
    "latitude": 14.8,
    "longitude": -3.38,
}
coordinate_store.loc["31"] = {
    "continent": "Africa",
    "country": "Mali",
    "type": "labyrinths",
    "latitude": 14.94,
    "longitude": -3.56,
}

# location with recent drought in Namibia (Issue #283)
coordinate_store.loc["32"] = {
    "continent": "Africa",
    "country": "Namibia",
    "type": "declining",
    "latitude": -18.05,
    "longitude": 12.76
}

# Baja
coordinate_store.loc["33"] = {
    "continent": "America",
    "country": "Mexico",
    "type": "labyrinths",
    "latitude": 26.77,
    "longitude": -112.92,
}

# Australia
coordinate_store.loc['34'] = {
  'continent' : 'Australia',
  'country': 'Australia',
  'type': 'labyrinths',
  'latitude': -23.35,
  'longitude': 133.36 # featured in June tech talk results
}
coordinate_store.loc['35'] = {
  'continent' : 'Australia',
  'country': 'Australia',
  'type': 'labyrinths',
  'latitude': -22.98,
  'longitude': 119.89
}


## Sites added by Josh:

#Australia
coordinate_store.loc['36'] = {
  'continent' : 'Australia', 
  'country': 'Australia', 
  'type': 'labyrinths', 
  'latitude': -25, 
  'longitude': 119.99
}

#Somalia
coordinate_store.loc['37'] = {
  'continent' : 'Africa', 
  'country': 'Somalia', 
  'type': 'labyrinths', 
  'latitude': 9.34, 
  'longitude': 48.64
}

coordinate_store.loc['38'] = {
  'continent' : 'Africa', 
  'country': 'Somalia', 
  'type': 'labyrinths', 
  'latitude': 9.63, 
  'longitude': 47.93
}


coordinate_store.loc['39'] = {
  'continent' : 'Africa', 
  'country': 'Somalia', 
  'type': 'labyrinths', 
  'latitude': 9.98, 
  'longitude': 48.44
}


coordinate_store.loc['40'] = {
  'continent' : 'Africa', 
  'country': 'Somalia', 
  'type': 'labyrinths', 
  'latitude': 4.64, 
  'longitude': 43.26
}


#Ethiopia
coordinate_store.loc['41'] = {
  'continent' : 'Africa', 
  'country': 'Ethiopia', 
  'type': 'Gaps', 
  'latitude': 4.69, 
  'longitude': 43.21
}


coordinate_store.loc['42'] = {
  'continent' : 'Africa', 
  'country': 'Ethiopia', 
  'type': 'labyrinths', 
  'latitude': 7.43, 
  'longitude': 42.9
}

#Kenya
coordinate_store.loc['43'] = {
  'continent' : 'Africa', 
  'country': 'Kenya', 
  'type': 'Gaps', 
  'latitude': 0.96, 
  'longitude': 40.37
}

#Mali
coordinate_store.loc['44'] = {
  'continent' : 'Africa', 
  'country': 'Mali', 
  'type': 'labyrinths', 
  'latitude': 14.80, 
  'longitude': -3.38
}

#Mexico
coordinate_store.loc['45'] = {
  'continent' : 'America', 
  'country': 'Mexico', 
  'type': 'labyrinths', 
  'latitude': 27.19, 
  'longitude': -103.92
}


#Chad
coordinate_store.loc["46"] = {
  'continent' : 'Africa', 
  'country': 'Chad', 
  'type': 'Gaps', 
  'latitude': 12, 
  'longitude': 19.99
}

coordinate_store.loc['47'] = {
  'continent' : 'Africa', 
  'country': 'Chad', 
  'type': 'Gaps', 
  'latitude': 12.05, 
  'longitude': 20.08
}

#Mali
coordinate_store.loc['48'] = {
  'continent' : 'Africa', 
  'country': 'Mali', 
  'type': 'labyrinths', 
  'latitude': 15.48, 
  'longitude': -5.83
}

#Mauritania
coordinate_store.loc['49'] = {
  'continent' : 'Africa', 
  'country': 'Mauritania', 
  'type': 'labyrinths', 
  'latitude': 15.57, 
  'longitude': -5.92
}

 
coordinate_store.loc['50'] = {
  'continent' : 'Africa', 
  'country': 'Mauritania', 
  'type': 'Gaps', 
  'latitude': 15.58, 
  'longitude': -13
}

#Nigeria
 
coordinate_store.loc['51'] = {
  'continent' : 'Africa', 
  'country': 'Nigeria', 
  'type': 'Gaps', 
  'latitude': 12.58, 
  'longitude': 3.75
}
 
#Niger 
coordinate_store.loc['52'] = {
  'continent' : 'Africa', 
  'country': 'Niger', 
  'type': 'Gaps', 
  'latitude': 12.7, 
  'longitude': 2.63
}
 
coordinate_store.loc['53'] = {
  'continent' : 'Africa', 
  'country': 'Niger', 
  'type': 'Gaps', 
  'latitude': 12.54, 
  'longitude': 2.26
}

coordinate_store.loc['54'] = {
  'continent' : 'Africa', 
  'country': 'Niger', 
  'type': 'Labyrinths', 
  'latitude': 13.12, 
  'longitude': 2.17
}

coordinate_store.loc['55'] = {
  'continent' : 'Africa', 
  'country': 'Sudan', 
  'type': 'Spots', 
  'latitude': 11.07, 
  'longitude': 27.93
}

coordinate_store.loc['56'] = {
  'continent' : 'Africa', 
  'country': 'Sudan', 
  'type': 'Spots', 
  'latitude': 11.30, 
  'longitude': 27.94
}

coordinate_store.loc['57'] = {
  'continent' : 'Africa', 
  'country': 'Sudan', 
  'type': 'Spots', 
  'latitude': 11.27, 
  'longitude': 27.55
}

coordinate_store.loc['58'] = {
  'continent' : 'Africa', 
  'country': 'Sudan', 
  'type': 'Spots', 
  'latitude': 11.47, 
  'longitude': 27.97
}

# hardcode a check to make sure we don't overwrite any rows

assert( len(coordinate_store) == 59 )
