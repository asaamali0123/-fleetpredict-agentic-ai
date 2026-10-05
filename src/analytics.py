import re
import numpy as np
import pandas as pd

THRESHOLDS={
    "engine_temp":(">",105,20),
    "oil_pressure":("<",30,25),
    "vibration":(">",7,20),
    "battery_voltage":("<",11.8,15),
}
ALIASES={
    "vehicle_id":["vehicle_id","vehicle","unit","truck_id","asset_id","vin"],
    "engine_temp":["engine_temp","engine_temperature","coolant_temp","engine_temp_c"],
    "oil_pressure":["oil_pressure","oil_press","oil_pressure_psi"],
    "vibration":["vibration","vibration_level","vibration_rms"],
    "battery_voltage":["battery_voltage","battery","battery_v","voltage"],
}

def norm(x): return re.sub(r"[^a-z0-9]+","_",str(x).lower()).strip("_")
def resolve_columns(df):
    n={norm(c):c for c in df.columns}; out={}
    for key,aliases in ALIASES.items():
        for a in aliases:
            if norm(a) in n: out[key]=n[norm(a)]; break
    return out

def vehicle_col(df):
    return resolve_columns(df).get("vehicle_id",df.columns[0])

def anomaly_row(row,cols):
    hits=[]; penalty=0
    for key,(op,thr,p) in THRESHOLDS.items():
        c=cols.get(key)
        if not c: continue
        v=pd.to_numeric(row.get(c),errors="coerce")
        if pd.isna(v): continue
        bad=v>thr if op==">" else v<thr
        if bad:
            hits.append(f"{key} {op} {thr} (observed {v:.2f})"); penalty+=p
    return hits,penalty

def score(p): return int(max(0,min(100,100-p)))
def level(s):
    if s>=90:return "Healthy"
    if s>=75:return "Watch"
    if s>=50:return "Warning"
    return "Critical"

def analyze_fleet(df):
    df=df.copy(); cols=resolve_columns(df); vc=vehicle_col(df); df[vc]=df[vc].astype(str)
    rows=[]; counts={k:0 for k in THRESHOLDS}
    for vid,g in df.groupby(vc,dropna=False):
        max_pen=0; hits=[]
        for _,row in g.iterrows():
            h,p=anomaly_row(row,cols); hits+=h; max_pen=max(max_pen,p)
        hs=score(max_pen)
        for key in THRESHOLDS:
            c=cols.get(key)
            if c:
                v=pd.to_numeric(g[c],errors="coerce"); op,thr,_=THRESHOLDS[key]
                counts[key]+=int(((v>thr) if op==">" else (v<thr)).sum())
        rows.append({"vehicle_id":str(vid),"health_score":hs,"risk_level":level(hs),
                     "risk_score":100-hs,"main_issue":"; ".join(hits[:2]) if hits else "No threshold anomaly",
                     "anomaly_count":len(hits)})
    summary=pd.DataFrame(rows).sort_values(["risk_score","anomaly_count"],ascending=False).reset_index(drop=True)
    return {
        "vehicle_summary":summary,
        "kpis":{"total":len(summary),"healthy":int((summary.risk_level=="Healthy").sum()),
                "warning_watch":int(summary.risk_level.isin(["Watch","Warning"]).sum()),
                "critical":int((summary.risk_level=="Critical").sum())},
        "anomaly_counts":pd.DataFrame({"metric":list(counts),"count":list(counts.values())}),
        "columns":cols,"rows":len(df)
    }

def vehicle_diagnostics(df,vehicle_id):
    a=analyze_fleet(df); cols=a["columns"]; vc=vehicle_col(df)
    g=df[df[vc].astype(str)==str(vehicle_id)]; row=g.iloc[-1]
    hits,p=anomaly_row(row,cols); hs=score(p)
    sensors={k:(None if pd.isna(row.get(c)) else round(float(row.get(c)),2)) for k,c in cols.items() if k!="vehicle_id"}
    if "oil_pressure" in " ".join(hits) and "vibration" in " ".join(hits):
        hint="Possible lubrication/mechanical issue pattern; verify oil system and rotating components."
    elif "engine_temp" in " ".join(hits):
        hint="Possible cooling-system or thermal-load issue; confirm coolant flow and operating conditions."
    elif hits: hint="Sensor anomaly detected; root cause is not confirmed by telemetry alone."
    else: hint="No immediate threshold-based fault pattern."
    priority="CRITICAL" if hs<50 else "HIGH" if hs<75 else "MEDIUM" if hs<90 else "LOW"
    return {"vehicle_id":str(vehicle_id),"health_score":hs,"risk_level":level(hs),"risk_score":100-hs,
            "sensors":sensors,"anomalies":hits,
            "evidence_summary":"; ".join(hits) if hits else "Configured thresholds are within bounds.",
            "diagnosis_hint":hint,"priority_action":priority}

def what_if(engine_temp,oil_pressure,vibration,battery_voltage):
    vals={"engine_temp":engine_temp,"oil_pressure":oil_pressure,"vibration":vibration,"battery_voltage":battery_voltage}
    hits=[]; p=0
    for k,(op,t,pen) in THRESHOLDS.items():
        v=vals[k]; bad=v>t if op==">" else v<t
        if bad: hits.append(f"{k} {op} {t} (observed {v:.2f})"); p+=pen
    hs=score(p)
    return {"health_score":hs,"risk_level":level(hs),"anomalies":hits}
