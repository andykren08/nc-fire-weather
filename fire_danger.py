import warnings
warnings.filterwarnings("ignore")

import os
import requests
from datetime import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES

# --- CONFIGURATION & THRESHOLDS ---
IMAGE_DIR = 'public/images'
PLOT_EXTENT = [-84.5, -75.0, 33.5, 37.0]  # Focused on North Carolina

# Red Flag / Critical Thresholds
CRITICAL_RH_MAX = 25.0        # %
CRITICAL_WIND_MIN = 20.0      # mph
CRITICAL_GUST_MIN = 30.0      # mph

# Elevated Fire Danger Thresholds (Provides a buffer category)
ELEVATED_RH_MAX = 30.0        # %
ELEVATED_WIND_MIN = 15.0      # mph
ELEVATED_GUST_MIN = 25.0      # mph

def get_domain_slice(ds, extent):
    lats = ds.latitude.values
    lons = ds.longitude.values
    lons = np.where(lons > 180, lons - 360, lons)
    
    mask = (
        (lons >= extent[0] - 1.0) & (lons <= extent[1] + 1.0) &
        (lats >= extent[2] - 1.0) & (lats <= extent[3] + 1.0)
    )
    
    rows, cols = np.where(mask)
    if len(rows) == 0: return slice(None), slice(None)

    pad = 5
    y_min, y_max = max(0, rows.min()-pad), min(lats.shape[0], rows.max()+pad)
    x_min, x_max = max(0, cols.min()-pad), min(lats.shape[1], cols.max()+pad)
    
    return slice(y_min, y_max), slice(x_min, x_max)

def calculate_rh(t_kelvin, d_kelvin):
    t_c = t_kelvin - 273.15
    d_c = d_kelvin - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * d_c) / (d_c + 243.5))
    return np.clip((e / es) * 100.0, 0, 100)

def ms_to_mph(speed_ms):
    return speed_ms * 2.23694

def calculate_fire_danger(rh, wind_mph, gust_mph):
    """
    Evaluates thresholds and returns a categorical grid:
    0 = None/Low, 1 = Elevated, 2 = Critical/Red Flag
    """
    danger_grid = np.zeros_like(rh, dtype=int)
    
    # Elevated Conditions
    elevated_mask = (rh <= ELEVATED_RH_MAX) & ((wind_mph >= ELEVATED_WIND_MIN) | (gust_mph >= ELEVATED_GUST_MIN))
    danger_grid[elevated_mask] = 1
    
    # Critical / Red Flag Conditions
    critical_mask = (rh <= CRITICAL_RH_MAX) & ((wind_mph >= CRITICAL_WIND_MIN) | (gust_mph >= CRITICAL_GUST_MIN))
    danger_grid[critical_mask] = 2
    
    return danger_grid

def download_file(url, local_filename):
    print(f"Downloading from: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename
    except Exception as e:
        print(f" -> Failed to download {local_filename}: {e}")
        return None

def generate_danger_plot(danger_grid, lats, lons, valid_time, fhr, run_str, model="HREF"):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-80, central_latitude=35))
    ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
    
    # Base Map Features
    ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black', zorder=10)
    try:
        ax.add_feature(USCOUNTIES.with_scale('5m'), linewidth=0.8, edgecolor='black', zorder=11, alpha=0.4)
    except: pass
    ax.add_feature(cfeature.OCEAN, facecolor='#cceeff')
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')

    # Custom Colormap for Fire Danger
    # 0: Transparent/None, 1: Orange (Elevated), 2: Magenta/Red (Critical)
    levels = [-0.5, 0.5, 1.5, 2.5]
    colors = ['#ffffff00', '#ffa500', '#d32f2f'] 
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, len(colors))

    mesh = ax.pcolormesh(lons, lats, danger_grid, cmap=cmap, norm=norm, 
                         transform=ccrs.PlateCarree(), shading='auto', zorder=5)

    # Time formatting
    t_str = str(valid_time).split('T')[1][:5]
    d_str = str(valid_time).split('T')[0]
    plt.title(f"{model} Fire Danger / Red Flag Potential\nValid: {d_str} {t_str}Z (F{fhr:02d})", loc='left', fontsize=12, fontweight='bold')
    plt.title(f"Run: {run_str}", loc='right', fontsize=10)
    
    cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.05, aspect=35, shrink=0.8)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Normal / None', 'Elevated', 'Critical (Red Flag)'])

    filename = f"{model.lower()}_fire_danger_f{fhr:02d}.png"
    save_path = os.path.join(IMAGE_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved {filename}")

def process_href(date_str, run_cycle, run_info):
    print("\n--- Processing HREF Fire Danger ---")
    global_lats, global_lons = None, None
    y_slice, x_slice = None, None 

    for fhr in range(1, 49):
        base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{date_str}/ensprod"
        filename = f"href.t{run_cycle}z.conus.mean.f{fhr:02d}.grib2"
        full_url = f"{base_url}/{filename}"
        
        grib = download_file(full_url, filename)
        if not grib: continue

        try:
            # Need T/Td for RH, and U/V/Gusts for wind
            ds_sfc = xr.open_dataset(grib, engine='cfgrib', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
            ds_wind = xr.open_dataset(grib, engine='cfgrib', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 10})
            ds_sfc_wind = xr.open_dataset(grib, engine='cfgrib', filter_by_keys={'typeOfLevel': 'surface'}) # Gusts often surface level in some models

            if global_lats is None:
                y_slice, x_slice = get_domain_slice(ds_sfc, PLOT_EXTENT)
                global_lats = ds_sfc.isel(y=y_slice, x=x_slice).latitude.values
                lons_raw = ds_sfc.isel(y=y_slice, x=x_slice).longitude.values
                global_lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)

            # Slicing
            ds_sfc_sub = ds_sfc.isel(y=y_slice, x=x_slice)
            ds_wind_sub = ds_wind.isel(y=y_slice, x=x_slice)
            
            # Variables Calculation
            rh = calculate_rh(ds_sfc_sub['t2m'].values, ds_sfc_sub['d2m'].values)
            
            # Wind Calculation (HREF usually provides u10 and v10 for 10m wind)
            u10 = ds_wind_sub['u10'].values
            v10 = ds_wind_sub['v10'].values
            wind_speed_ms = np.sqrt(u10**2 + v10**2)
            wind_mph = ms_to_mph(wind_speed_ms)
            
            # Extract gusts (try surface first, fallback to 10m)
            try:
                ds_gust_sub = xr.open_dataset(grib, engine='cfgrib', filter_by_keys={'stepType': 'max', 'shortName': 'gust'}).isel(y=y_slice, x=x_slice)
                gust_mph = ms_to_mph(ds_gust_sub['gust'].values)
                ds_gust_sub.close()
            except:
                # If gust variable isn't found easily, default to sustained wind to prevent crashing
                gust_mph = wind_mph

            # Calculate Threat Grid
            danger_grid = calculate_fire_danger(rh, wind_mph, gust_mph)
            
            # Plot
            generate_danger_plot(danger_grid, global_lats, global_lons, ds_sfc_sub.valid_time.values, fhr, run_info, model="HREF")

            ds_sfc.close()
            ds_wind.close()
            ds_sfc_wind.close()

        except Exception as e:
            print(f"Error f{fhr:02d}: {e}")
        finally:
            if os.path.exists(filename): os.remove(filename)

def process_ndfd():
    print("\n--- Processing NDFD Comparison ---")
    ndfd_wspd_url = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.wspd.bin"
    ndfd_gust_url = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.gust.bin"
    ndfd_rh_url = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.rhm.bin"
    
    wspd_file = download_file(ndfd_wspd_url, "ndfd_wspd.grib2")
    gust_file = download_file(ndfd_gust_url, "ndfd_gust.grib2")
    rh_file = download_file(ndfd_rh_url, "ndfd_rh.grib2")

    if wspd_file and gust_file and rh_file:
        try:
            ds_wspd = xr.open_dataset(wspd_file, engine='cfgrib')
            ds_gust = xr.open_dataset(gust_file, engine='cfgrib')
            ds_rh = xr.open_dataset(rh_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '2r'}})

            n_ysl, n_xsl = get_domain_slice(ds_rh, PLOT_EXTENT)
            ds_wspd_sub = ds_wspd.isel(y=n_ysl, x=n_xsl)
            ds_gust_sub = ds_gust.isel(y=n_ysl, x=n_xsl)
            ds_rh_sub = ds_rh.isel(y=n_ysl, x=n_xsl)
            
            n_lats = ds_rh_sub.latitude.values
            n_lons = ds_rh_sub.longitude.values
            n_lons = np.where(n_lons > 180, n_lons - 360, n_lons)

            try:
                ndfd_time_np = ds_rh_sub.time.values
                ts = (ndfd_time_np - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                ndfd_init_dt = datetime.utcfromtimestamp(ts)
                ndfd_run_info = ndfd_init_dt.strftime("%Y%m%d %HZ")
            except Exception:
                ndfd_run_info = "Operational NDFD"

            valid_times_rh = np.atleast_1d(ds_rh_sub.valid_time.values)
            valid_times_wspd = np.atleast_1d(ds_wspd_sub.valid_time.values)
            
            # Find times where we have both RH and Wind
            common_times = np.intersect1d(valid_times_rh, valid_times_wspd)

            fhr = 1
            for v_time in np.sort(common_times):
                if fhr > 48: break 
                
                rh_idx = np.where(valid_times_rh == v_time)[0][0]
                wspd_idx = np.where(valid_times_wspd == v_time)[0][0]
                
                rh_data = ds_rh_sub.isel(step=rh_idx)['r2'].values if 'r2' in ds_rh_sub.data_vars else ds_rh_sub.isel(step=rh_idx)['2r'].values
                wspd_ms = ds_wspd_sub.isel(step=wspd_idx)['10si'].values if '10si' in ds_wspd_sub.data_vars else ds_wspd_sub.isel(step=wspd_idx)['wspd'].values
                wind_mph = ms_to_mph(wspd_ms)
                
                # Fetch Gust if valid time aligns, otherwise assume sustained = gust
                gust_mph = wind_mph
                if v_time in ds_gust_sub.valid_time.values:
                    g_idx = np.where(ds_gust_sub.valid_time.values == v_time)[0][0]
                    gust_ms = ds_gust_sub.isel(step=g_idx)['gust'].values
                    gust_mph = ms_to_mph(gust_ms)

                danger_grid = calculate_fire_danger(rh_data, wind_mph, gust_mph)
                generate_danger_plot(danger_grid, n_lats, n_lons, v_time, fhr, ndfd_run_info, model="NDFD")
                
                fhr += 1
                
            ds_wspd.close()
            ds_gust.close()
            ds_rh.close()
        except Exception as e:
            print(f"Error processing NDFD: {e}")
        finally:
            for f in [wspd_file, gust_file, rh_file]:
                if os.path.exists(f): os.remove(f)

def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    now = datetime.utcnow()
    run_cycle = "12" if now.hour >= 14 else "00"
    date_str = now.strftime("%Y%m%d")
    run_info = f"{date_str} {run_cycle}Z"
    
    # Process HREF Model
    process_href(date_str, run_cycle, run_info)
    
    # Process Operational NDFD for comparison
    process_ndfd()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    main()
