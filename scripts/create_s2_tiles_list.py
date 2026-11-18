import geopandas as gpd
import matplotlib.pyplot as plt

KML = "S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml"
COUNTRIES = "ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

tiles = gpd.read_file(KML, engine="pyogrio", layer="Features")
tiles = tiles.explode(index_parts=False)
tiles = tiles[tiles.geom_type == "Polygon"].set_crs(4326, allow_override=True)

world = gpd.read_file(COUNTRIES)
germany = world[world["ADMIN"] == "Germany"].to_crs(4326)
ger_geom = germany.geometry.union_all()

hits = tiles[tiles.intersects(ger_geom)][["Name", "geometry"]].drop_duplicates("Name")
hits = hits.sort_values("Name")

# GeoJSON outputs
hits.to_file("s2_tiles_germany.geojson", driver="GeoJSON")
germany.to_file("germany.geojson", driver="GeoJSON")

# tile ID list
with open("s2_tiles_germany.txt", "w") as f:
    f.write("\n".join(hits["Name"]))

print(len(hits), "tiles")

# quick visual check
ax = hits.plot(alpha=0.4, edgecolor="k", figsize=(9, 11))
germany.boundary.plot(ax=ax, color="red", linewidth=1)
for _, r in hits.iterrows():
    c = r.geometry.centroid
    ax.annotate(r["Name"], (c.x, c.y), ha="center", fontsize=6)
plt.savefig("s2_tiles_germany.png", dpi=200, bbox_inches="tight")
