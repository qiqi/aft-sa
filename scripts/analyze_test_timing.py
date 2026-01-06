#!/usr/bin/env python3
"""
Analyze test timing reports.

Usage:
    # Generate timing report first:
    pytest tests/ --timing-report
    
    # Then analyze:
    python scripts/analyze_test_timing.py tests/timing_report.json
    
    # Compare two runs:
    python scripts/analyze_test_timing.py tests/timing_report.json --compare old_report.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def load_report(path: Path) -> Dict[str, Any]:
    """Load timing report from JSON."""
    with open(path) as f:
        return json.load(f)


def format_duration(seconds: float) -> str:
    """Format duration with appropriate units."""
    if seconds >= 60:
        return f"{seconds/60:.1f}m"
    elif seconds >= 1:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds*1000:.1f}ms"


def print_summary(report: Dict[str, Any]):
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print(f"TEST TIMING ANALYSIS - {report['timestamp']}")
    print(f"{'='*70}")
    
    print(f"\n📊 Overview:")
    print(f"   Total time:  {format_duration(report['total_duration'])}")
    print(f"   Tests:       {report['test_count']} total")
    print(f"   ✅ Passed:    {report['passed']}")
    print(f"   ❌ Failed:    {report['failed']}")
    print(f"   ⏭️  Skipped:   {report['skipped']}")
    
    # Calculate percentiles
    tests = report['tests']
    if tests:
        durations = sorted([t['duration'] for t in tests])
        n = len(durations)
        p50 = durations[n//2]
        p90 = durations[int(n*0.9)]
        p99 = durations[int(n*0.99)] if n >= 100 else durations[-1]
        
        print(f"\n📈 Duration Percentiles:")
        print(f"   Median (p50):  {format_duration(p50)}")
        print(f"   p90:           {format_duration(p90)}")
        print(f"   p99:           {format_duration(p99)}")
        print(f"   Max:           {format_duration(durations[-1])}")


def print_slowest_tests(report: Dict[str, Any], n: int = 20):
    """Print slowest tests."""
    print(f"\n🐢 Top {n} Slowest Tests:")
    print(f"   {'Duration':<12} {'Test'}")
    print(f"   {'-'*12} {'-'*55}")
    
    for test in report['tests'][:n]:
        duration = format_duration(test['duration'])
        # Shorten nodeid for display
        nodeid = test['nodeid']
        if len(nodeid) > 55:
            nodeid = "..." + nodeid[-52:]
        print(f"   {duration:<12} {nodeid}")


def print_module_breakdown(report: Dict[str, Any]):
    """Print timing by module."""
    print(f"\n📁 Module Breakdown:")
    print(f"   {'Duration':<12} {'Tests':<8} {'Avg':<10} {'Module'}")
    print(f"   {'-'*12} {'-'*8} {'-'*10} {'-'*35}")
    
    for mod, data in report['modules'].items():
        duration = format_duration(data['total_duration'])
        avg = format_duration(data['avg_duration'])
        # Shorten module path
        mod_short = mod.replace("tests/", "").replace(".py", "")
        if len(mod_short) > 35:
            mod_short = "..." + mod_short[-32:]
        print(f"   {duration:<12} {data['test_count']:<8} {avg:<10} {mod_short}")


def print_class_breakdown(report: Dict[str, Any], n: int = 15):
    """Print timing by test class."""
    if not report['classes']:
        return
    
    print(f"\n🏷️  Top {n} Test Classes by Duration:")
    print(f"   {'Duration':<12} {'Tests':<8} {'Class'}")
    print(f"   {'-'*12} {'-'*8} {'-'*45}")
    
    for i, (cls, data) in enumerate(report['classes'].items()):
        if i >= n:
            break
        duration = format_duration(data['total_duration'])
        # Shorten class path
        cls_short = cls.replace("tests/", "").replace(".py", "")
        if len(cls_short) > 45:
            cls_short = "..." + cls_short[-42:]
        print(f"   {duration:<12} {data['test_count']:<8} {cls_short}")


def print_fixture_breakdown(report: Dict[str, Any], n: int = 10):
    """Print timing by fixture."""
    if not report['fixtures']:
        return
    
    print(f"\n🔧 Top {n} Fixtures by Total Time:")
    print(f"   {'Total':<12} {'Calls':<8} {'Avg':<12} {'Max':<12} {'Fixture'}")
    print(f"   {'-'*12} {'-'*8} {'-'*12} {'-'*12} {'-'*25}")
    
    for i, (name, data) in enumerate(report['fixtures'].items()):
        if i >= n:
            break
        total = format_duration(data['total_duration'])
        avg = format_duration(data['avg_duration'])
        max_d = format_duration(data['max_duration'])
        print(f"   {total:<12} {data['call_count']:<8} {avg:<12} {max_d:<12} {name}")


def compare_reports(current: Dict[str, Any], baseline: Dict[str, Any]):
    """Compare two timing reports."""
    print(f"\n{'='*70}")
    print("COMPARISON WITH BASELINE")
    print(f"{'='*70}")
    
    # Overall comparison
    curr_dur = current['total_duration']
    base_dur = baseline['total_duration']
    diff = curr_dur - base_dur
    pct = (diff / base_dur) * 100 if base_dur > 0 else 0
    
    symbol = "🔺" if diff > 0 else "🔻" if diff < 0 else "⏸️"
    print(f"\n📊 Overall: {format_duration(curr_dur)} vs {format_duration(base_dur)} ({symbol} {pct:+.1f}%)")
    
    # Find regressions and improvements
    curr_tests = {t['nodeid']: t for t in current['tests']}
    base_tests = {t['nodeid']: t for t in baseline['tests']}
    
    regressions = []
    improvements = []
    
    for nodeid, curr in curr_tests.items():
        if nodeid in base_tests:
            base = base_tests[nodeid]
            diff = curr['duration'] - base['duration']
            pct = (diff / base['duration']) * 100 if base['duration'] > 0 else 0
            
            if pct > 20 and diff > 0.1:  # >20% slower and >100ms difference
                regressions.append((nodeid, curr['duration'], base['duration'], pct))
            elif pct < -20 and diff < -0.1:  # >20% faster
                improvements.append((nodeid, curr['duration'], base['duration'], pct))
    
    if regressions:
        print(f"\n⚠️  Regressions (>20% slower):")
        for nodeid, curr_d, base_d, pct in sorted(regressions, key=lambda x: x[3], reverse=True)[:10]:
            print(f"   {format_duration(curr_d)} vs {format_duration(base_d)} ({pct:+.1f}%)  {nodeid[-50:]}")
    
    if improvements:
        print(f"\n✅ Improvements (>20% faster):")
        for nodeid, curr_d, base_d, pct in sorted(improvements, key=lambda x: x[3])[:10]:
            print(f"   {format_duration(curr_d)} vs {format_duration(base_d)} ({pct:+.1f}%)  {nodeid[-50:]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze test timing reports")
    parser.add_argument("report", type=Path, help="Path to timing_report.json")
    parser.add_argument("--compare", type=Path, help="Baseline report to compare against")
    parser.add_argument("--top", type=int, default=20, help="Number of slowest tests to show")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text")
    
    args = parser.parse_args()
    
    report = load_report(args.report)
    
    if args.json:
        print(json.dumps(report, indent=2))
        return
    
    print_summary(report)
    print_slowest_tests(report, args.top)
    print_module_breakdown(report)
    print_class_breakdown(report)
    print_fixture_breakdown(report)
    
    if args.compare:
        baseline = load_report(args.compare)
        compare_reports(report, baseline)
    
    print()


if __name__ == "__main__":
    main()

