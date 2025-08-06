#!/usr/bin/env python3
"""
CAST.ai Organization Analysis Tool

This script performs comprehensive analysis of CAST.ai clusters including:
1. Cost and efficiency analysis
2. Workload autoscaler analysis

Usage:
    export CAST_AI_API_KEY="your_api_key_here"
    python cast_ai_analyzer.py

Requirements:
    pip install requests pandas python-dateutil numpy

Author: Organization Analysis Tool
License: MIT
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import time
import json
from typing import List, Dict, Tuple, Optional
import sys

def get_api_key() -> str:
    """Get API key from environment variable"""
    api_key = os.getenv('CAST_AI_API_KEY')
    if not api_key:
        print("❌ Error: CAST_AI_API_KEY environment variable not set")
        print("💡 Please set your API key: export CAST_AI_API_KEY='your_api_key_here'")
        sys.exit(1)
    return api_key

def get_all_clusters(api_key: str) -> List[Dict]:
    """Fetch all clusters for the organization"""
    url = "https://api.eu.cast.ai/v1/kubernetes/external-clusters"
    headers = {
        "X-API-Key": api_key,
        "accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        clusters = response.json().get("items", [])
        print(f"🌐 Found {len(clusters)} clusters in organization")
        return clusters
    except Exception as e:
        print(f"❌ Error fetching clusters: {e}")
        return []

def get_cluster_dates(api_key: str, cluster_id: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get cluster creation and first operation dates"""
    url = f"https://api.eu.cast.ai/v1/kubernetes/external-clusters/{cluster_id}"
    headers = {
        "X-API-Key": api_key,
        "accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        created_at = datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
        first_operation_at = datetime.fromisoformat(data["firstOperationAt"].replace("Z", "+00:00"))
        return created_at, first_operation_at
    except Exception as e:
        print(f"⚠️ Error getting dates for cluster {cluster_id}: {e}")
        return None, None

def generate_monthly_locks(start_date: datetime, end_date: Optional[datetime] = None) -> List[Dict]:
    """Generate monthly date locks for API calls"""
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    locks = []
    current = start_date.replace(day=1)
    while current < end_date:
        next_month = current + relativedelta(months=1)
        lock_start = current
        lock_end = min(next_month - timedelta(seconds=1), end_date)
        locks.append({
            "start": lock_start,
            "end": lock_end
        })
        current = next_month
    return locks

def get_baseline_efficiency_summary(api_key: str, cluster_id: str, start: datetime, end: datetime) -> Dict:
    """Get baseline efficiency summary for OP factor calculation"""
    start = start.replace(minute=0, second=0, microsecond=0)
    end = end.replace(minute=0, second=0, microsecond=0)
    start_str = start.isoformat().replace("+00:00", "Z")
    end_str = end.isoformat().replace("+00:00", "Z")

    url = (
        f"https://api.eu.cast.ai/v1/cost-reports/clusters/{cluster_id}/efficiency"
        f"?startTime={start_str}&endTime={end_str}&stepSeconds=3600"
    )
    headers = {
        "X-API-Key": api_key,
        "accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        summary = data.get("summary", {})
        items = data.get("items", [])

        if items:
            df = pd.DataFrame(items)

            # Convert columns to numeric
            numeric_cols = [
                "cpuCountOnDemand", "cpuCountSpot", "cpuCountSpotFallback",
                "requestedCpuCountOnDemand", "requestedCpuCountSpot", "requestedCpuCountSpotFallback",
                "ramGibOnDemand", "ramGibSpot", "ramGibSpotFallback",
                "requestedRamGibOnDemand", "requestedRamGibSpot", "requestedRamGibSpotFallback",
                "storageProvisionedGib", "requestedStorageGib", "storageClaimedGib",
                "cpuCostOnDemand", "cpuCostSpot", "cpuCostSpotFallback",
                "ramCostOnDemand", "ramCostSpot", "ramCostSpotFallback"
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Compute totals
            df["total_provisioned_cpu"] = df["cpuCountOnDemand"] + df["cpuCountSpot"] + df["cpuCountSpotFallback"]
            df["total_requested_cpu"] = df["requestedCpuCountOnDemand"] + df["requestedCpuCountSpot"] + df["requestedCpuCountSpotFallback"]
            df["total_cpu_cost"] = df["cpuCostOnDemand"] + df["cpuCostSpot"] + df["cpuCostSpotFallback"]
            df["total_provisioned_ram"] = df["ramGibOnDemand"] + df["ramGibSpot"] + df["ramGibSpotFallback"]
            df["total_requested_ram"] = df["requestedRamGibOnDemand"] + df["requestedRamGibSpot"] + df["requestedRamGibSpotFallback"]
            df["total_ram_cost"] = df["ramCostOnDemand"] + df["ramCostSpot"] + df["ramCostSpotFallback"]

            # Add computed values to summary
            summary["avgProvisionedCpu"] = round(df["total_provisioned_cpu"].mean(), 6)
            summary["avgRequestedCpu"] = round(df["total_requested_cpu"].mean(), 6)
            summary["avgProvisionedRamGib"] = round(df["total_provisioned_ram"].mean(), 6)
            summary["avgRequestedRamGib"] = round(df["total_requested_ram"].mean(), 6)
            summary["avgCpuCost"] = round(df["total_cpu_cost"].mean(), 6)
            summary["avgRamCost"] = round(df["total_ram_cost"].mean(), 6)

            # Calculate cost per provisioned unit (hourly)
            if summary["avgProvisionedCpu"] > 0:
                summary["costPerProvisionedCpu"] = round(summary["avgCpuCost"] / summary["avgProvisionedCpu"], 6)
            else:
                summary["costPerProvisionedCpu"] = None

            if summary["avgProvisionedRamGib"] > 0:
                summary["costPerProvisionedRamGib"] = round(summary["avgRamCost"] / summary["avgProvisionedRamGib"], 6)
            else:
                summary["costPerProvisionedRamGib"] = None

            # Calculate overprovisioning factors
            if summary["avgRequestedCpu"] > 0:
                summary["cpuOpFactor"] = round(summary["avgProvisionedCpu"] / summary["avgRequestedCpu"], 3)
            else:
                summary["cpuOpFactor"] = None

            if summary["avgRequestedRamGib"] > 0:
                summary["ramOpFactor"] = round(summary["avgProvisionedRamGib"] / summary["avgRequestedRamGib"], 3)
            else:
                summary["ramOpFactor"] = None

        return summary
    except Exception as e:
        print(f"⚠️ Error getting baseline summary for cluster {cluster_id}: {e}")
        return {}

def get_workload_metrics_daily_avg(
    cluster_id: str,
    api_key: str,
    from_time: str,
    to_time: str,
    base_url: str = "https://api.eu.cast.ai/v1"
) -> pd.DataFrame:
    """Fetch workload autoscaling metrics from Cast.ai API and return daily averages"""
    
    url = f"{base_url}/workload-autoscaling/clusters/{cluster_id}/workloads-summary-metrics"
    
    headers = {
        'X-API-Key': api_key,
        'accept': 'application/json'
    }
    
    params = {
        'fromTime': from_time,
        'toTime': to_time
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'items' not in data:
            raise ValueError("Response does not contain 'items' key")
        
        items = data['items']
        
        if not items:
            return pd.DataFrame()
        
        df = pd.DataFrame(items)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        numeric_columns = [
            'cpuRequestCores',
            'memoryRequestGibs',
            'cpuOriginalRequestCores',
            'memoryOriginalRequestGibs',
            'cpuRecommendationCores',
            'memoryRecommendationGibs',
            'cpuUsageCores',
            'memoryUsageGibs'
        ]
        
        # Group by date and calculate daily averages
        daily_avg = df.groupby('date')[numeric_columns].mean().reset_index()
        
        # Round to 3 decimal places
        for col in numeric_columns:
            daily_avg[col] = daily_avg[col].round(3)
        
        daily_avg['date'] = pd.to_datetime(daily_avg['date'])
        
        return daily_avg
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")

def fetch_workload_metrics_for_locks(api_key: str, cluster_id: str, locks: List[Dict]) -> pd.DataFrame:
    """Fetch workload metrics for all date locks and combine"""
    all_workload_dfs = []

    for i, lock in enumerate(locks):
        print(f"    📅 Fetching workload metrics period {i+1}/{len(locks)}: {lock['start']} to {lock['end']}")
        
        start_str = lock['start'].isoformat().replace("+00:00", "Z")
        end_str = lock['end'].isoformat().replace("+00:00", "Z")
        
        try:
            workload_df = get_workload_metrics_daily_avg(
                cluster_id=cluster_id,
                api_key=api_key,
                from_time=start_str,
                to_time=end_str
            )
            
            if not workload_df.empty:
                all_workload_dfs.append(workload_df)
                
        except Exception as e:
            print(f"      ⚠️ Error fetching workload metrics for {lock['start']} - {lock['end']}: {e}")
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)

    if all_workload_dfs:
        full_workload_df = pd.concat(all_workload_dfs, ignore_index=True)
        full_workload_df = full_workload_df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        return full_workload_df
    else:
        return pd.DataFrame()

def add_baseline_analysis_columns(workload_df: pd.DataFrame, baseline_summary: Dict) -> pd.DataFrame:
    """Add analysis columns based on baseline summary"""
    
    cpu_op_factor = baseline_summary.get("cpuOpFactor", 1.0) or 1.0
    ram_op_factor = baseline_summary.get("ramOpFactor", 1.0) or 1.0
    cost_per_provisioned_cpu = baseline_summary.get("costPerProvisionedCpu", 0.0) or 0.0
    cost_per_provisioned_ram = baseline_summary.get("costPerProvisionedRamGib", 0.0) or 0.0
    
    # Calculate differences between current and original requests
    workload_df['cpuRequestDiff'] = (workload_df['cpuOriginalRequestCores'] - workload_df['cpuRequestCores']).round(3)
    workload_df['memoryRequestDiff'] = (workload_df['memoryOriginalRequestGibs'] - workload_df['memoryRequestGibs']).round(3)
    
    # Calculate how much provisioned CPU/Memory would be required for the request difference
    workload_df['cpuProvisionedForDiff'] = (workload_df['cpuRequestDiff'] * cpu_op_factor).round(3)
    workload_df['memoryProvisionedForDiff'] = (workload_df['memoryRequestDiff'] * ram_op_factor).round(3)
    
    # Calculate cost for provisioned differences (hourly)
    workload_df['cpuProvisionedForDiffCostHourly'] = (workload_df['cpuProvisionedForDiff'] * cost_per_provisioned_cpu).round(6)
    workload_df['memoryProvisionedForDiffCostHourly'] = (workload_df['memoryProvisionedForDiff'] * cost_per_provisioned_ram).round(6)
    
    # Calculate daily cost for provisioned differences (24 hours)
    workload_df['cpuProvisionedForDiffCostDaily'] = (workload_df['cpuProvisionedForDiffCostHourly'] * 24).round(4)
    workload_df['memoryProvisionedForDiffCostDaily'] = (workload_df['memoryProvisionedForDiffCostHourly'] * 24).round(4)
    
    # Calculate total daily cost savings from optimization
    workload_df['totalProvisionedDiffCostDaily'] = (workload_df['cpuProvisionedForDiffCostDaily'] + workload_df['memoryProvisionedForDiffCostDaily']).round(4)
    
    # Calculate total provisioned requirements based on current requests
    workload_df['cpuProvisionedCurrent'] = (workload_df['cpuRequestCores'] * cpu_op_factor).round(3)
    workload_df['memoryProvisionedCurrent'] = (workload_df['memoryRequestGibs'] * ram_op_factor).round(3)
    
    # Calculate total provisioned requirements based on original requests
    workload_df['cpuProvisionedOriginal'] = (workload_df['cpuOriginalRequestCores'] * cpu_op_factor).round(3)
    workload_df['memoryProvisionedOriginal'] = (workload_df['memoryOriginalRequestGibs'] * ram_op_factor).round(3)
    
    # Calculate daily costs for current and original provisioned resources
    workload_df['cpuProvisionedCurrentCostDaily'] = (workload_df['cpuProvisionedCurrent'] * cost_per_provisioned_cpu * 24).round(4)
    workload_df['memoryProvisionedCurrentCostDaily'] = (workload_df['memoryProvisionedCurrent'] * cost_per_provisioned_ram * 24).round(4)
    workload_df['cpuProvisionedOriginalCostDaily'] = (workload_df['cpuProvisionedOriginal'] * cost_per_provisioned_cpu * 24).round(4)
    workload_df['memoryProvisionedOriginalCostDaily'] = (workload_df['memoryProvisionedOriginal'] * cost_per_provisioned_ram * 24).round(4)
    
    # Calculate total daily costs
    workload_df['totalProvisionedCurrentCostDaily'] = (workload_df['cpuProvisionedCurrentCostDaily'] + workload_df['memoryProvisionedCurrentCostDaily']).round(4)
    workload_df['totalProvisionedOriginalCostDaily'] = (workload_df['cpuProvisionedOriginalCostDaily'] + workload_df['memoryProvisionedOriginalCostDaily']).round(4)
    
    # Calculate efficiency metrics (avoid division by zero)
    workload_df['cpuUtilizationPercent'] = np.where(
        workload_df['cpuRequestCores'] > 0,
        ((workload_df['cpuUsageCores'] / workload_df['cpuRequestCores']) * 100).round(2),
        0
    )
    workload_df['memoryUtilizationPercent'] = np.where(
        workload_df['memoryRequestGibs'] > 0,
        ((workload_df['memoryUsageGibs'] / workload_df['memoryRequestGibs']) * 100).round(2),
        0
    )
    
    # Calculate recommendation vs current request ratios
    workload_df['cpuRecommendationRatio'] = np.where(
        workload_df['cpuRequestCores'] > 0,
        (workload_df['cpuRecommendationCores'] / workload_df['cpuRequestCores']).round(3),
        0
    )
    workload_df['memoryRecommendationRatio'] = np.where(
        workload_df['memoryRequestGibs'] > 0,
        (workload_df['memoryRecommendationGibs'] / workload_df['memoryRequestGibs']).round(3),
        0
    )
    
    # Add baseline factors and costs as columns for reference
    workload_df['baselineCpuOpFactor'] = cpu_op_factor
    workload_df['baselineRamOpFactor'] = ram_op_factor
    workload_df['baselineCostPerProvisionedCpu'] = cost_per_provisioned_cpu
    workload_df['baselineCostPerProvisionedRam'] = cost_per_provisioned_ram
    
    return workload_df

def fetch_report(url: str, api_key: str) -> List[Dict]:
    """Fetch report data from API"""
    headers = {
        "X-API-Key": api_key,
        "accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get("items", [])
    except Exception as e:
        print(f"⚠️ Error fetching report: {e}")
        return []

def fetch_cost_and_efficiency(api_key: str, cluster_id: str, start: datetime, end: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch cost and efficiency data for a time period"""
    start = start.replace(minute=0, second=0, microsecond=0)
    end = end.replace(minute=0, second=0, microsecond=0)
    start_str = start.isoformat().replace("+00:00", "Z")
    end_str = end.isoformat().replace("+00:00", "Z")

    base_url = f"https://api.eu.cast.ai/v1/cost-reports/clusters/{cluster_id}"
    cost_url = f"{base_url}/cost?startTime={start_str}&endTime={end_str}&stepSeconds=3600"
    efficiency_url = f"{base_url}/efficiency?startTime={start_str}&endTime={end_str}&stepSeconds=3600"

    try:
        cost_items = fetch_report(cost_url, api_key)
        efficiency_items = fetch_report(efficiency_url, api_key)

        cost_df = pd.DataFrame(cost_items)
        eff_df = pd.DataFrame(efficiency_items)

        if not cost_df.empty:
            cost_df["timestamp"] = pd.to_datetime(cost_df["timestamp"])
        if not eff_df.empty:
            eff_df["timestamp"] = pd.to_datetime(eff_df["timestamp"])

        return cost_df, eff_df

    except Exception as e:
        print(f"⚠️ Error fetching data for {start} - {end}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def build_full_dataframes(api_key: str, cluster_id: str, locks: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build complete dataframes from multiple time periods"""
    all_cost_dfs = []
    all_eff_dfs = []

    for i, lock in enumerate(locks):
        print(f"    📅 Fetching cost/efficiency period {i+1}/{len(locks)}: {lock['start']} to {lock['end']}")
        cost_df, eff_df = fetch_cost_and_efficiency(api_key, cluster_id, lock['start'], lock['end'])

        if not cost_df.empty:
            all_cost_dfs.append(cost_df)
        if not eff_df.empty:
            all_eff_dfs.append(eff_df)
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)

    full_cost_df = pd.concat(all_cost_dfs, ignore_index=True) if all_cost_dfs else pd.DataFrame()
    full_eff_df = pd.concat(all_eff_dfs, ignore_index=True) if all_eff_dfs else pd.DataFrame()
    
    return full_cost_df, full_eff_df

def calculate_organization_baseline_average(baseline_summaries: List[Dict]) -> Dict:
    """Calculate average baseline factors from successful clusters"""
    if not baseline_summaries:
        return {}
    
    print(f"\n📊 Calculating organization average baseline from {len(baseline_summaries)} clusters...")
    
    # Collect all valid baseline factors
    cpu_op_factors = []
    ram_op_factors = []
    cpu_costs = []
    ram_costs = []
    
    for baseline in baseline_summaries:
        if baseline.get("cpuOpFactor") and not np.isnan(baseline["cpuOpFactor"]):
            cpu_op_factors.append(baseline["cpuOpFactor"])
        if baseline.get("ramOpFactor") and not np.isnan(baseline["ramOpFactor"]):
            ram_op_factors.append(baseline["ramOpFactor"])
        if baseline.get("costPerProvisionedCpu") and not np.isnan(baseline["costPerProvisionedCpu"]):
            cpu_costs.append(baseline["costPerProvisionedCpu"])
        if baseline.get("costPerProvisionedRamGib") and not np.isnan(baseline["costPerProvisionedRamGib"]):
            ram_costs.append(baseline["costPerProvisionedRamGib"])
    
    # Calculate averages
    avg_baseline = {}
    if cpu_op_factors:
        avg_baseline["cpuOpFactor"] = round(sum(cpu_op_factors) / len(cpu_op_factors), 3)
    if ram_op_factors:
        avg_baseline["ramOpFactor"] = round(sum(ram_op_factors) / len(ram_op_factors), 3)
    if cpu_costs:
        avg_baseline["costPerProvisionedCpu"] = round(sum(cpu_costs) / len(cpu_costs), 6)
    if ram_costs:
        avg_baseline["costPerProvisionedRamGib"] = round(sum(ram_costs) / len(ram_costs), 6)
    
    # Add additional required fields with computed averages
    if cpu_op_factors and cpu_costs:
        avg_baseline["avgProvisionedCpu"] = 1.0  # Placeholder
        avg_baseline["avgRequestedCpu"] = 1.0 / avg_baseline["cpuOpFactor"]  # Derived
        avg_baseline["avgCpuCost"] = avg_baseline["costPerProvisionedCpu"]  # Same as cost per unit
    
    if ram_op_factors and ram_costs:
        avg_baseline["avgProvisionedRamGib"] = 1.0  # Placeholder
        avg_baseline["avgRequestedRamGib"] = 1.0 / avg_baseline["ramOpFactor"]  # Derived
        avg_baseline["avgRamCost"] = avg_baseline["costPerProvisionedRamGib"]  # Same as cost per unit
    
    print(f"✅ Organization average baseline calculated:")
    print(f"   CPU OP Factor: {avg_baseline.get('cpuOpFactor', 'N/A')}")
    print(f"   RAM OP Factor: {avg_baseline.get('ramOpFactor', 'N/A')}")
    print(f"   CPU Cost/Unit: ${avg_baseline.get('costPerProvisionedCpu', 0):.6f}")
    print(f"   RAM Cost/Unit: ${avg_baseline.get('costPerProvisionedRamGib', 0):.6f}")
    
    return avg_baseline

def safe_divide(numerator, denominator):
    """Safe division with NaN handling"""
    return np.where(denominator != 0, numerator / denominator, np.nan)

def has_sufficient_baseline_data(baseline_start: datetime, baseline_end: datetime, baseline_summary: Dict) -> bool:
    """Check if cluster has sufficient baseline data"""
    # Check if we have at least 7 days of baseline period for cost analysis
    baseline_days = (baseline_end - baseline_start).days
    has_enough_days = baseline_days >= 7
    
    # Check if we have valid OP factors
    has_valid_cpu_factor = baseline_summary.get("cpuOpFactor") is not None and not np.isnan(baseline_summary.get("cpuOpFactor", np.nan))
    has_valid_ram_factor = baseline_summary.get("ramOpFactor") is not None and not np.isnan(baseline_summary.get("ramOpFactor", np.nan))
    
    return has_enough_days and has_valid_cpu_factor and has_valid_ram_factor

def has_sufficient_workload_baseline_data(baseline_start: datetime, baseline_end: datetime, baseline_summary: Dict) -> bool:
    """Check if cluster has sufficient baseline data for workload analysis (14 days)"""
    # Check if we have at least 14 days of baseline period for workload analysis
    baseline_days = (baseline_end - baseline_start).days
    has_enough_days = baseline_days >= 14
    
    # Check if we have valid OP factors and cost data
    has_valid_cpu_factor = baseline_summary.get("cpuOpFactor") is not None and not np.isnan(baseline_summary.get("cpuOpFactor", np.nan))
    has_valid_ram_factor = baseline_summary.get("ramOpFactor") is not None and not np.isnan(baseline_summary.get("ramOpFactor", np.nan))
    has_cpu_cost = baseline_summary.get("costPerProvisionedCpu") is not None and not np.isnan(baseline_summary.get("costPerProvisionedCpu", np.nan))
    has_ram_cost = baseline_summary.get("costPerProvisionedRamGib") is not None and not np.isnan(baseline_summary.get("costPerProvisionedRamGib", np.nan))
    
    return has_enough_days and has_valid_cpu_factor and has_valid_ram_factor and has_cpu_cost and has_ram_cost

def process_single_cluster_cost_efficiency(api_key: str, cluster_info: Dict, output_dir: str = "cost_outputs", 
                                         fallback_baseline: Optional[Dict] = None) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Process cost and efficiency data for a single cluster"""
    cluster_id = cluster_info["id"]
    cluster_name = cluster_info.get("name", f"cluster_{cluster_id[:8]}")
    
    print(f"\n  🔄 Processing cost & efficiency for: {cluster_name} ({cluster_id})")
    
    # Create output directory for this cluster
    cluster_output_dir = os.path.join(output_dir, f"{cluster_name}_{cluster_id[:8]}")
    os.makedirs(cluster_output_dir, exist_ok=True)
    
    # Get cluster dates
    created_at, first_operation_at = get_cluster_dates(api_key, cluster_id)
    if not created_at or not first_operation_at:
        print(f"    ⚠️ Skipping cluster {cluster_name} - couldn't get dates")
        return None, None
    
    print(f"    📅 Cluster created: {created_at}")
    print(f"    📅 First operation: {first_operation_at}")
    
    # Calculate data start date (max 7 days before first operation)
    max_data_start = first_operation_at - timedelta(days=7)
    data_start_date = max(created_at, max_data_start)
    print(f"    📅 Data fetching from: {data_start_date}")
    
    # Generate locks and fetch data
    locks = generate_monthly_locks(data_start_date)
    cost_df, eff_df = build_full_dataframes(api_key, cluster_id, locks)
    
    if cost_df.empty and eff_df.empty:
        print(f"    ⚠️ No data found for cluster {cluster_name}")
        return None, None
    
    # Get baseline efficiency summary
    baseline_start = max(created_at, first_operation_at - timedelta(days=7))
    baseline_end = first_operation_at
    baseline_summary = get_baseline_efficiency_summary(api_key, cluster_id, baseline_start, baseline_end)
    
    # Check if we have sufficient baseline data
    has_sufficient_baseline = has_sufficient_baseline_data(baseline_start, baseline_end, baseline_summary)
    
    baseline_source = "cluster_specific"
    if not has_sufficient_baseline:
        if fallback_baseline:
            print(f"    ⚠️ Insufficient baseline data for {cluster_name} - using organization average")
            baseline_summary = fallback_baseline.copy()
            baseline_source = "organization_average"
        else:
            print(f"    ⚠️ Insufficient baseline data for {cluster_name} - will retry in second pass")
            return None, None
    else:
        print(f"    ✅ Using cluster-specific baseline - CPU: {baseline_summary.get('cpuOpFactor')}, RAM: {baseline_summary.get('ramOpFactor')}")
    
    # Process cost data
    daily_cost = pd.DataFrame()
    if not cost_df.empty:
        cost_df["date"] = cost_df["timestamp"].dt.date
        for col in cost_df.columns:
            if col not in ["timestamp", "date"]:
                cost_df[col] = pd.to_numeric(cost_df[col], errors="coerce")
        
        daily_cost = cost_df.groupby("date").sum(numeric_only=True).reset_index()
        
        # Create combined cost metrics
        combined_cost = pd.DataFrame({
            'date': daily_cost['date'],
            'totalCost': daily_cost['costOnDemand'] + daily_cost['costSpot'] + daily_cost['costSpotFallback'],
            'totalCpuCount': daily_cost['cpuCountOnDemand'] + daily_cost['cpuCountSpot'] + daily_cost['cpuCountSpotFallback'],
            'totalCpuCost': daily_cost['cpuCostOnDemand'] + daily_cost['cpuCostSpot'] + daily_cost['cpuCostSpotFallback'],
            'totalRamCost': daily_cost['ramCostOnDemand'] + daily_cost['ramCostSpot'] + daily_cost['ramCostSpotFallback'],
            'totalRamGib': daily_cost['ramGibOnDemand'] + daily_cost['ramGibSpot'] + daily_cost['ramGibSpotFallback'],
            'totalGpuCost': daily_cost['gpuCostOnDemand'] + daily_cost['gpuCostSpot'] + daily_cost['gpuCostSpotFallback'],
            'totalGpuCount': daily_cost['gpuCountOnDemand'] + daily_cost['gpuCountSpot'] + daily_cost['gpuCountSpotFallback'],
            'totalStorageGib': daily_cost['storageGib'],
            'totalStorageCost': daily_cost['storageCost']
        })
    
    # Process efficiency data
    daily_eff = pd.DataFrame()
    if not eff_df.empty:
        eff_df["date"] = eff_df["timestamp"].dt.date
        for col in eff_df.columns:
            if col not in ["timestamp", "date"]:
                eff_df[col] = pd.to_numeric(eff_df[col], errors="coerce")
        
        daily_eff = eff_df.groupby("date").mean(numeric_only=True).reset_index()
        
        # Create combined efficiency metrics
        combined_eff = pd.DataFrame({
            'date': daily_eff['date'],
            'totalCpuCost': daily_eff['cpuCostOnDemand'] + daily_eff['cpuCostSpot'] + daily_eff['cpuCostSpotFallback'],
            'totalRamCost': daily_eff['ramCostOnDemand'] + daily_eff['ramCostSpot'] + daily_eff['ramCostSpotFallback'],
            'totalCpuCount': daily_eff['cpuCountOnDemand'] + daily_eff['cpuCountSpot'] + daily_eff['cpuCountSpotFallback'],
            'totalRamGib': daily_eff['ramGibOnDemand'] + daily_eff['ramGibSpot'] + daily_eff['ramGibSpotFallback'],
            'totalRequestedCpuCount': daily_eff['requestedCpuCountOnDemand'] + daily_eff['requestedCpuCountSpot'] + daily_eff['requestedCpuCountSpotFallback'],
            'totalRequestedRamGib': daily_eff['requestedRamGibOnDemand'] + daily_eff['requestedRamGibSpot'] + daily_eff['requestedRamGibSpotFallback'],
            'totalCpuUsed': daily_eff['cpuUsedOnDemand'] + daily_eff['cpuUsedSpot'] + daily_eff['cpuUsedSpotFallback'],
            'totalRamUsedGib': daily_eff['ramUsedGibOnDemand'] + daily_eff['ramUsedGibSpot'] + daily_eff['ramUsedGibSpotFallback'],
            'totalCpuOverprovisioning': daily_eff['cpuOverprovisioningOnDemand'] + daily_eff['cpuOverprovisioningSpot'] + daily_eff['cpuOverprovisioningSpotFallback'],
            'totalRamOverprovisioning': daily_eff['ramOverprovisioningOnDemand'] + daily_eff['ramOverprovisioningSpot'] + daily_eff['ramOverprovisioningSpotFallback'],
            'storageProvisionedGib': daily_eff['storageProvisionedGib'],
            'storageClaimedGib': daily_eff['storageClaimedGib'],
            'requestedStorageGib': daily_eff['requestedStorageGib'],
            'storageCost': daily_eff['storageCost']
        })
    
    # Combine cost and efficiency data
    if not combined_cost.empty and not combined_eff.empty:
        combined_df = pd.merge(combined_cost, combined_eff, on="date", how="outer", suffixes=('_cost', '_eff'))
    elif not combined_cost.empty:
        combined_df = combined_cost.copy()
    elif not combined_eff.empty:
        combined_df = combined_eff.copy()
    else:
        print(f"    ⚠️ No combined data for cluster {cluster_name}")
        return None, None
    
    # Calculate baseline metrics and projections
    if not combined_df.empty and baseline_summary:
        # Convert date to datetime for filtering
        combined_df['date'] = pd.to_datetime(combined_df['date']).dt.tz_localize('UTC')
        baseline_df = combined_df[(combined_df['date'] >= baseline_start) & (combined_df['date'] <= baseline_end)]
        
        if not baseline_df.empty and baseline_source == "cluster_specific":
            # Calculate average costs per unit from actual cluster data
            cpu_cost_col = 'totalCpuCost_cost' if 'totalCpuCost_cost' in baseline_df.columns else 'totalCpuCost'
            cpu_count_col = 'totalCpuCount_cost' if 'totalCpuCount_cost' in baseline_df.columns else 'totalCpuCount'
            ram_cost_col = 'totalRamCost_cost' if 'totalRamCost_cost' in baseline_df.columns else 'totalRamCost'
            ram_gib_col = 'totalRamGib_cost' if 'totalRamGib_cost' in baseline_df.columns else 'totalRamGib'
            
            avg_cost_per_cpu = safe_divide(baseline_df[cpu_cost_col], baseline_df[cpu_count_col]).mean()
            avg_cost_per_ram = safe_divide(baseline_df[ram_cost_col], baseline_df[ram_gib_col]).mean()
        else:
            # Use default cost estimates when using fallback baseline
            avg_cost_per_cpu = 0.04  # Default estimate: $0.04 per CPU hour
            avg_cost_per_ram = 0.01  # Default estimate: $0.01 per GB RAM hour
        
        # Get overprovisioning factors
        cpu_op_factor = baseline_summary.get("cpuOpFactor", 1.0) or 1.0
        ram_op_factor = baseline_summary.get("ramOpFactor", 1.0) or 1.0
        
        # Calculate projected metrics
        if 'totalRequestedCpuCount' in combined_df.columns:
            combined_df['projectedCpu'] = combined_df['totalRequestedCpuCount'] * cpu_op_factor
            combined_df['projectedCpuCost'] = combined_df['projectedCpu'] * avg_cost_per_cpu * 24
        
        if 'totalRequestedRamGib' in combined_df.columns:
            combined_df['projectedRam'] = combined_df['totalRequestedRamGib'] * ram_op_factor
            combined_df['projectedRamCost'] = combined_df['projectedRam'] * avg_cost_per_ram * 24
    
    # Add cluster identification and baseline source
    combined_df['cluster_id'] = cluster_id
    combined_df['cluster_name'] = cluster_name
    combined_df['baseline_source'] = baseline_source
    
    # Save individual cluster results
    output_path = os.path.join(cluster_output_dir, f"{cluster_name}_cost_efficiency.csv")
    combined_df.to_csv(output_path, index=False)
    print(f"    ✅ Cost & efficiency data saved: {output_path}")
    
    # Return baseline summary only if it's cluster-specific (for organization average calculation)
    return combined_df, baseline_summary if has_sufficient_baseline and baseline_source == "cluster_specific" else None

def process_single_cluster_workload(api_key: str, cluster_info: Dict, output_dir: str = "workload_outputs", 
                                  fallback_baseline: Optional[Dict] = None) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Process workload metrics for a single cluster, return (dataframe, baseline_summary)"""
    cluster_id = cluster_info["id"]
    cluster_name = cluster_info.get("name", f"cluster_{cluster_id[:8]}")
    
    print(f"\n  🔄 Processing workload metrics for: {cluster_name} ({cluster_id})")
    
    # Create output directory for this cluster
    cluster_output_dir = os.path.join(output_dir, f"{cluster_name}_{cluster_id[:8]}")
    os.makedirs(cluster_output_dir, exist_ok=True)
    
    # Get cluster dates
    created_at, first_operation_at = get_cluster_dates(api_key, cluster_id)
    if not created_at or not first_operation_at:
        print(f"    ⚠️ Skipping cluster {cluster_name} - couldn't get dates")
        return None, None
    
    print(f"    📅 Cluster created: {created_at}")
    print(f"    📅 First operation: {first_operation_at}")
    
    # Calculate data start date (max 14 days before first operation)
    max_data_start = first_operation_at - timedelta(days=14)
    data_start_date = max(created_at, max_data_start)
    print(f"    📅 Data fetching from: {data_start_date}")
    
    # Generate date locks
    locks = generate_monthly_locks(data_start_date)
    print(f"    📊 Generated {len(locks)} monthly locks")
    
    # Get baseline efficiency summary
    baseline_start = max(created_at, first_operation_at - timedelta(days=14))
    baseline_end = first_operation_at
    
    print(f"    📊 Getting baseline efficiency summary ({baseline_start} to {baseline_end})...")
    baseline_summary = get_baseline_efficiency_summary(api_key, cluster_id, baseline_start, baseline_end)
    
    # Check if baseline data is sufficient for workload analysis
    has_sufficient_baseline = has_sufficient_workload_baseline_data(baseline_start, baseline_end, baseline_summary)
    
    baseline_source = "cluster_specific"
    if not has_sufficient_baseline:
        if fallback_baseline:
            print(f"    ⚠️ Insufficient baseline data for {cluster_name}, using organization average baseline")
            baseline_summary = fallback_baseline.copy()
            baseline_source = 'organization_average'
        else:
            print(f"    ⚠️ Insufficient baseline data for {cluster_name} and no fallback available - will retry in second pass")
            return None, None
    else:
        print(f"    ✅ Baseline OP Factors - CPU: {baseline_summary.get('cpuOpFactor')}, RAM: {baseline_summary.get('ramOpFactor')}")
    
    # Fetch workload metrics
    print("    📈 Fetching workload metrics...")
    workload_df = fetch_workload_metrics_for_locks(api_key, cluster_id, locks)
    
    if workload_df.empty:
        print(f"    ⚠️ No workload metrics data for cluster {cluster_name}")
        return None, baseline_summary if has_sufficient_baseline and baseline_source == "cluster_specific" else None
    
    print(f"    ✅ Retrieved workload metrics for {len(workload_df)} days")
    
    # Add baseline analysis columns
    print("    🔬 Adding baseline analysis columns...")
    enhanced_workload_df = add_baseline_analysis_columns(workload_df, baseline_summary)
    
    # Add cluster identification and baseline source
    enhanced_workload_df['cluster_id'] = cluster_id
    enhanced_workload_df['cluster_name'] = cluster_name
    enhanced_workload_df['baseline_source'] = baseline_source
    
    # Save individual cluster results
    output_path = os.path.join(cluster_output_dir, f"{cluster_name}_workload_metrics.csv")
    enhanced_workload_df.to_csv(output_path, index=False)
    print(f"    ✅ Workload data saved: {output_path}")
    
    return enhanced_workload_df, baseline_summary if has_sufficient_baseline and baseline_source == "cluster_specific" else None

def run_cost_efficiency_analysis(api_key: str, clusters: List[Dict], main_output_dir: str) -> Tuple[List[pd.DataFrame], List[Dict]]:
    """Run cost and efficiency analysis for all clusters"""
    print(f"\n{'='*80}")
    print("💰 STARTING COST & EFFICIENCY ANALYSIS")
    print(f"{'='*80}")
    
    cost_output_dir = os.path.join(main_output_dir, "cost_efficiency_analysis")
    os.makedirs(cost_output_dir, exist_ok=True)
    
    all_cost_data = []
    successful_clusters = []
    failed_clusters = []
    baseline_summaries = []
    clusters_needing_fallback = []
    
    print(f"\n🔄 FIRST PASS: Processing clusters with sufficient baseline data")
    
    # First pass: Process clusters with sufficient baseline data
    for i, cluster in enumerate(clusters):
        print(f"\n--- Cost analysis - First pass: Cluster {i+1}/{len(clusters)} ---")
        
        try:
            cluster_df, baseline_summary = process_single_cluster_cost_efficiency(
                api_key, cluster, cost_output_dir, fallback_baseline=None
            )
            
            if cluster_df is not None:
                all_cost_data.append(cluster_df)
                successful_clusters.append(cluster)
                
                if baseline_summary:  # This means it had sufficient baseline data
                    baseline_summaries.append(baseline_summary)
                    
                print(f"  ✅ Successfully processed: {cluster.get('name', cluster['id'])}")
            else:
                clusters_needing_fallback.append(cluster)
                print(f"  ⏳ Will retry in second pass: {cluster.get('name', cluster['id'])}")
                
        except Exception as e:
            print(f"  ❌ Error processing cluster {cluster.get('name', cluster['id'])}: {e}")
            failed_clusters.append(cluster)
        
        # Add delay between clusters to avoid rate limiting
        time.sleep(1)
    
    # Calculate organization average baseline from successful clusters
    org_avg_baseline = calculate_organization_baseline_average(baseline_summaries)
    
    if clusters_needing_fallback and org_avg_baseline:
        print(f"\n🔄 SECOND PASS: Processing {len(clusters_needing_fallback)} clusters with fallback baseline")
        
        # Second pass: Process clusters using organization average baseline
        for i, cluster in enumerate(clusters_needing_fallback):
            print(f"\n--- Cost analysis - Second pass: Cluster {i+1}/{len(clusters_needing_fallback)} ---")
            
            try:
                cluster_df, _ = process_single_cluster_cost_efficiency(
                    api_key, cluster, cost_output_dir, fallback_baseline=org_avg_baseline
                )
                
                if cluster_df is not None:
                    all_cost_data.append(cluster_df)
                    successful_clusters.append(cluster)
                    print(f"  ✅ Successfully processed with fallback: {cluster.get('name', cluster['id'])}")
                else:
                    failed_clusters.append(cluster)
                    print(f"  ❌ Failed even with fallback: {cluster.get('name', cluster['id'])}")
                    
            except Exception as e:
                print(f"  ❌ Error processing cluster {cluster.get('name', cluster['id'])}: {e}")
                failed_clusters.append(cluster)
            
            # Add delay between clusters to avoid rate limiting
            time.sleep(1)
    
    elif clusters_needing_fallback and not org_avg_baseline:
        print(f"\n⚠️ {len(clusters_needing_fallback)} clusters need fallback baseline, but no successful baselines available")
        failed_clusters.extend(clusters_needing_fallback)
    
    # Consolidate cost analysis results
    if all_cost_data:
        print(f"\n📊 CONSOLIDATING COST & EFFICIENCY RESULTS")
        
        master_cost_df = pd.concat(all_cost_data, ignore_index=True)
        master_cost_df = master_cost_df.sort_values(['cluster_name', 'date']).reset_index(drop=True)
        
        # Save master CSV
        master_csv_path = os.path.join(cost_output_dir, "master_organization_cost_efficiency.csv")
        master_cost_df.to_csv(master_csv_path, index=False)
        
        print(f"✅ Cost & efficiency master CSV saved: {master_csv_path}")
        print(f"📊 Total cost records: {len(master_cost_df)}")
        print(f"📊 Successful clusters: {len(successful_clusters)}")
        
        if failed_clusters:
            print(f"⚠️ Failed clusters: {len(failed_clusters)}")
    
    return all_cost_data, baseline_summaries

def run_workload_analysis(api_key: str, clusters: List[Dict], main_output_dir: str, 
                         baseline_summaries: List[Dict]) -> List[pd.DataFrame]:
    """Run workload autoscaler analysis for all clusters"""
    print(f"\n{'='*80}")
    print("🚀 STARTING WORKLOAD AUTOSCALER ANALYSIS")
    print(f"{'='*80}")
    
    workload_output_dir = os.path.join(main_output_dir, "workload_analysis")
    os.makedirs(workload_output_dir, exist_ok=True)
    
    all_workload_data = []
    successful_clusters = []
    failed_clusters = []
    clusters_needing_fallback = []
    
    print(f"\n🔄 FIRST PASS: Processing clusters with sufficient baseline data")
    
    # First pass: Process clusters with sufficient baseline data
    for i, cluster in enumerate(clusters):
        print(f"\n--- Workload analysis - First pass: Cluster {i+1}/{len(clusters)} ---")
        
        try:
            workload_df, baseline_summary = process_single_cluster_workload(
                api_key, cluster, workload_output_dir, fallback_baseline=None
            )
            
            if workload_df is not None:
                all_workload_data.append(workload_df)
                successful_clusters.append(cluster)
                print(f"  ✅ Successfully processed: {cluster.get('name', cluster['id'])}")
            else:
                clusters_needing_fallback.append(cluster)
                print(f"  ⏳ Will retry in second pass: {cluster.get('name', cluster['id'])}")
                
        except Exception as e:
            print(f"  ❌ Error processing cluster {cluster.get('name', cluster['id'])}: {e}")
            failed_clusters.append(cluster)
        
        # Add delay between clusters to avoid rate limiting
        time.sleep(1)
    
    # Calculate organization average baseline from successful clusters (or use from cost analysis)
    org_avg_baseline = calculate_organization_baseline_average(baseline_summaries)
    
    if clusters_needing_fallback and org_avg_baseline:
        print(f"\n🔄 SECOND PASS: Processing {len(clusters_needing_fallback)} clusters with fallback baseline")
        
        # Second pass: Process clusters using organization average baseline
        for i, cluster in enumerate(clusters_needing_fallback):
            print(f"\n--- Workload analysis - Second pass: Cluster {i+1}/{len(clusters_needing_fallback)} ---")
            
            try:
                workload_df, _ = process_single_cluster_workload(
                    api_key, cluster, workload_output_dir, fallback_baseline=org_avg_baseline
                )
                
                if workload_df is not None:
                    all_workload_data.append(workload_df)
                    successful_clusters.append(cluster)
                    print(f"  ✅ Successfully processed with fallback: {cluster.get('name', cluster['id'])}")
                else:
                    failed_clusters.append(cluster)
                    print(f"  ❌ Failed even with fallback: {cluster.get('name', cluster['id'])}")
                    
            except Exception as e:
                print(f"  ❌ Error processing cluster {cluster.get('name', cluster['id'])}: {e}")
                failed_clusters.append(cluster)
            
            # Add delay between clusters to avoid rate limiting
            time.sleep(1)
    
    elif clusters_needing_fallback and not org_avg_baseline:
        print(f"\n⚠️ {len(clusters_needing_fallback)} clusters need fallback baseline, but no successful baselines available")
        failed_clusters.extend(clusters_needing_fallback)
    
    # Consolidate workload analysis results
    if all_workload_data:
        print(f"\n📊 CONSOLIDATING WORKLOAD RESULTS")
        
        master_workload_df = pd.concat(all_workload_data, ignore_index=True)
        master_workload_df = master_workload_df.sort_values(['cluster_name', 'date']).reset_index(drop=True)
        
        # Save master CSV
        master_csv_path = os.path.join(workload_output_dir, "master_organization_workload_metrics.csv")
        master_workload_df.to_csv(master_csv_path, index=False)
        
        print(f"✅ Workload master CSV saved: {master_csv_path}")
        print(f"📊 Total workload records: {len(master_workload_df)}")
        print(f"📊 Successful clusters: {len(successful_clusters)}")
        
        # Calculate organization-wide workload summary
        total_daily_savings = master_workload_df['totalProvisionedDiffCostDaily'].sum()
        avg_cpu_utilization = master_workload_df['cpuUtilizationPercent'].mean()
        avg_memory_utilization = master_workload_df['memoryUtilizationPercent'].mean()
        
        print(f"💰 Organization daily savings: ${total_daily_savings:,.2f}")
        print(f"💰 Estimated monthly savings: ${total_daily_savings * 30:,.2f}")
        print(f"⚡ Average CPU utilization: {avg_cpu_utilization:.1f}%")
        print(f"🧠 Average memory utilization: {avg_memory_utilization:.1f}%")
        
        if failed_clusters:
            print(f"⚠️ Failed clusters: {len(failed_clusters)}")
    
    return all_workload_data

def main():
    """Main execution function"""
    print("🚀 CAST.ai Organization Analysis Tool")
    print("=" * 50)
    
    # Get API key from environment
    api_key = get_api_key()
    
    # Create main output directory
    main_output_dir = "cast_ai_organization_analysis"
    os.makedirs(main_output_dir, exist_ok=True)
    
    print("🌐 Starting organization-wide analysis...")
    
    # Get all clusters
    clusters = get_all_clusters(api_key)
    if not clusters:
        print("❌ No clusters found. Exiting.")
        return
    
    # Run cost and efficiency analysis first
    cost_data, baseline_summaries = run_cost_efficiency_analysis(api_key, clusters, main_output_dir)
    
    # Run workload autoscaler analysis
    workload_data = run_workload_analysis(api_key, clusters, main_output_dir, baseline_summaries)
    
    # Create final summary report
    print(f"\n{'='*80}")
    print("📋 CREATING FINAL SUMMARY REPORT")
    print(f"{'='*80}")
    
    summary_report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_clusters_found': len(clusters),
        'cost_analysis': {
            'successful_clusters': len(cost_data) if cost_data else 0,
            'total_records': sum(len(df) for df in cost_data) if cost_data else 0
        },
        'workload_analysis': {
            'successful_clusters': len(workload_data) if workload_data else 0,
            'total_records': sum(len(df) for df in workload_data) if workload_data else 0
        },
        'organization_baseline_factors': calculate_organization_baseline_average(baseline_summaries) if baseline_summaries else {},
        'baseline_summary': {
            'clusters_with_sufficient_baseline': len(baseline_summaries),
            'fallback_cpu_op_factor': calculate_organization_baseline_average(baseline_summaries).get('cpuOpFactor') if baseline_summaries else None,
            'fallback_ram_op_factor': calculate_organization_baseline_average(baseline_summaries).get('ramOpFactor') if baseline_summaries else None
        }
    }
    
    # Add workload-specific metrics if available
    if workload_data:
        master_workload_df = pd.concat(workload_data, ignore_index=True)
        total_daily_savings = master_workload_df['totalProvisionedDiffCostDaily'].sum()
        avg_cpu_utilization = master_workload_df['cpuUtilizationPercent'].mean()
        avg_memory_utilization = master_workload_df['memoryUtilizationPercent'].mean()
        
        summary_report['workload_metrics'] = {
            'total_daily_cost_savings': round(total_daily_savings, 2),
            'avg_cpu_utilization_percent': round(avg_cpu_utilization, 2),
            'avg_memory_utilization_percent': round(avg_memory_utilization, 2),
            'total_monthly_savings_estimate': round(total_daily_savings * 30, 2)
        }
    
    # Save summary report
    summary_path = os.path.join(main_output_dir, "organization_analysis_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    print(f"📋 Final analysis summary saved: {summary_path}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print("🎉 ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"🌐 Total clusters found: {len(clusters)}")
    print(f"💰 Cost analysis completed for: {len(cost_data) if cost_data else 0} clusters")
    print(f"🚀 Workload analysis completed for: {len(workload_data) if workload_data else 0} clusters")
    
    if workload_data:
        master_workload_df = pd.concat(workload_data, ignore_index=True)
        total_daily_savings = master_workload_df['totalProvisionedDiffCostDaily'].sum()
        print(f"💰 Total daily savings from workload optimization: ${total_daily_savings:,.2f}")
        print(f"💰 Estimated monthly savings: ${total_daily_savings * 30:,.2f}")
    
    print(f"\n📁 All results saved in: {main_output_dir}/")
    print("✅ Analysis completed successfully!")

if __name__ == "__main__":
    main()
