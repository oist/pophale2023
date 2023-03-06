import pandas as pd
import h5py

def map_id2name(ids, lut):
    id2name = {}
    for _, row in lut.iterrows():
        id2name[row["ID"]] = row["region_name2"]
    id2name[0] = "NaN"
    return [id2name[i] for i in ids]

def map_id2color(ids, lut):
    out = []
    for i in ids:
        row = lut[lut["ID"] == i]
        c = row["region_color"].values.tolist()
        if c:
            out.append([int(x) for x in str(c[0]).split(",")])
        else:
            out.append([0,0,0])
    return out

def merge_data_oct(data, hdf_info, atlas_lut):
    probes = pd.DataFrame()

    for (j, d) in enumerate(data):
        df = pd.read_csv(d["channel_lut"])
        df = df.loc[100-d["offset"]:] # crop extrapolated channels
        df["X"] /= 10 # convert um to pixel
        df["Y"] /= 10
        df["Z"] /= 10
        df["Region_name"] = map_id2name(df["Region"].to_numpy(), atlas_lut)
        df["Region_color"] = map_id2color(df["Region"].to_numpy(), atlas_lut)
        
        for h in hdf_info:
            with h5py.File(h["path"]) as hf:
                df[h["name"]] = hf[h["key"]][0:df.shape[0], d["hdf5_column"]]
        
        df["probe"] = d["name"]

        probes = pd.concat([probes, df], ignore_index=True, sort=False)
    
    return probes
