
import googlemaps
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
import pandas as pd
import csv


df = pd.read_csv("tweets_smell_LOC_after_trainedNER_modified.csv")
print(len(df))


location= list(df["LOC"])
# print(type(location[1]))
for i in range(len(location)):
    if location[i] == np.nan:
        location[i] = None
    else:
        location[i] = str(location[i]) + ", Singapore"
print(location)


# print(location[0])

gmaps_key = googlemaps.Client(key = "AIzaSyAoGOEKGg229cqFRGDb15kPezLH-wnoZ_E")
geocode_result = gmaps_key.geocode(location[15])
print(location[15])
print(geocode_result[0])

lats = []
lons = []
addresses = []
tweets=[]
processed_tweets=[]
index =[]
id=[]
LOC=[]
place=[]
for i in range(0, len(location), 1):
    geocode_result = gmaps_key.geocode(location[i])
    try:
        lat = geocode_result[0]["geometry"]["location"]["lat"]
        lon = geocode_result[0]["geometry"]["location"]["lng"]
        address = location[i]
        lats.append(lat)
        lons.append(lon)
        addresses.append(address)
        loc=df['LOC'].iloc[i]
        LOC.append(loc)
        tweet=df['tweet'].iloc[i]
        tweets.append(tweet)
        processed_tweet=df['processed_tweets'].iloc[i]
        processed_tweets.append(processed_tweet)
        ind= df['all_triggers'].iloc[i]
        index.append(ind)
        ids=df['id'].iloc[i]
        id.append(ids)
        p=df['place'].iloc[i]
        place.append(p)


    except:
        lat = None
        lon = None

dicts={"location":addresses, "LOC":LOC, "place":place,  "lon":lons,"lat":lats, "id":id,"tweet":tweets,"processed_tweets":processed_tweets,"index":index}

locations = pd.DataFrame(dicts)
# locations['id'] = df['id']
# locations['LOC'] = df['LOC']
# locations['tweet'] = df['tweet']
# locations['processed_tweets'] = df['processed_tweets']
# locations['index']=df['all_triggers']
# locations['date'] = df['date']
# locations['time'] = df['time']
# locations['place'] = df['place']


locations.to_csv('LOC_locations.csv',index=False)

# overpass_url = "http://overpass-api.de/api/interpreter"
# overpass_query = """
# [out:json];
# area[name="London"][type=boundary][boundary=administrative];
# (node["amenity"="pub"](area);
#  way["amenity"="pub"](area);
#  rel["amenity"="pub"](area);
# );
# out center;
# """
# response = requests.get(overpass_url,
#                         params={'data': overpass_query})
# data = response.json()
#
# print(data)