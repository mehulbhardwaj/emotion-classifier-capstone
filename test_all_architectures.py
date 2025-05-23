#!/usr/bin/env python3
"""
Test all three emotion classification architectures and compare their performance.

This script trains MLP Fusion, DialogRNN, and TOD-KAT models on MELD dataset
and provides a comparison of their F1 scores and training characteristics.
"""

import os
import subprocess
import time
import yaml
from pathlib import Path
from datetime import datetime


def run_training(config_path, architecture_name):
    """Run training for a specific architecture and capture results."""
    print(f"\n🚀 Starting training for {architecture_name.upper()}")
    print(f"📋 Config: {config_path}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Run training
        result = subprocess.run(
            ["python", "train.py", "--config", config_path],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {architecture_name.upper()} training completed successfully!")
            print(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            
            # Extract key metrics from output
            lines = result.stdout.split('\n')
            best_val_f1 = None
            test_f1 = None
            total_params = None
            trainable_params = None
            
            for line in lines:
                if "val_f1" in line and "epoch" in line:
                    # Extract validation F1 from checkpoint filename
                    parts = line.split("val_f1=")
                    if len(parts) > 1:
                        val_f1_str = parts[1].split("}")[0]
                        try:
                            best_val_f1 = float(val_f1_str)
                        except:
                            pass
                elif "test_f1" in line and "=" in line:
                    # Extract test F1
                    parts = line.split("test_f1=")
                    if len(parts) > 1:
                        test_f1_str = parts[1].split()[0]
                        try:
                            test_f1 = float(test_f1_str)
                        except:
                            pass
                elif "Total params" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "params" and i > 0:
                            try:
                                total_params = parts[i-1]
                            except:
                                pass
                elif "Trainable params" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "params" and i > 0:
                            try:
                                trainable_params = parts[i-1]
                            except:
                                pass
            
            return {
                'success': True,
                'duration': duration,
                'best_val_f1': best_val_f1,
                'test_f1': test_f1,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"❌ {architecture_name.upper()} training failed!")
            print(f"Error output:\n{result.stderr}")
            return {
                'success': False,
                'duration': duration,
                'error': result.stderr,
                'stdout': result.stdout
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {architecture_name.upper()} training timed out after 1 hour")
        return {
            'success': False,
            'duration': 3600,
            'error': 'Training timed out after 1 hour'
        }
    except Exception as e:
        print(f"💥 {architecture_name.upper()} training crashed: {e}")
        return {
            'success': False,
            'duration': time.time() - start_time,
            'error': str(e)
        }


def main():
    """Run all architecture tests and compare results."""
    print("🔬 EMOTION CLASSIFICATION ARCHITECTURE COMPARISON")
    print("="*60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define architectures to test
    architectures = [
        {
            'name': 'mlp_fusion',
            'config': 'configs/colab_config_mlp_fusion.yaml',
            'description': 'Simple MLP fusion baseline'
        },
        {
            'name': 'dialog_rnn',
            'config': 'configs/colab_config_dialog_rnn.yaml', 
            'description': 'DialogRNN with context and speaker modeling'
        },
        {
            'name': 'todkat_lite',
            'config': 'configs/colab_config_todkat_lite.yaml',
            'description': 'TOD-KAT with topic modeling and attention'
        }
    ]
    
    results = {}
    
    # Test each architecture
    for arch in architectures:
        print(f"\n📍 Testing {arch['name']}: {arch['description']}")
        
        # Check if config exists
        if not os.path.exists(arch['config']):
            print(f"❌ Config file not found: {arch['config']}")
            results[arch['name']] = {
                'success': False,
                'error': f"Config file not found: {arch['config']}"
            }
            continue
            
        # Run training
        result = run_training(arch['config'], arch['name'])
        results[arch['name']] = result
        
        # Brief pause between runs
        if result['success']:
            print(f"✅ {arch['name']} completed - pausing 30 seconds before next run...")
            time.sleep(30)
    
    # Print summary report
    print("\n" + "="*60)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*60)
    
    successful_runs = []
    failed_runs = []
    
    for arch_name, result in results.items():
        if result['success']:
            successful_runs.append((arch_name, result))
            print(f"\n✅ {arch_name.upper()}:")
            print(f"   Duration: {result['duration']:.1f}s ({result['duration']/60:.1f}min)")
            print(f"   Trainable Params: {result.get('trainable_params', 'N/A')}")
            print(f"   Total Params: {result.get('total_params', 'N/A')}")
            print(f"   Best Val F1: {result.get('best_val_f1', 'N/A')}")
            print(f"   Test F1: {result.get('test_f1', 'N/A')}")
        else:
            failed_runs.append((arch_name, result))
            print(f"\n❌ {arch_name.upper()}: FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Performance comparison
    if len(successful_runs) >= 2:
        print(f"\n🏆 PERFORMANCE RANKING:")
        print("="*40)
        
        # Sort by test F1 if available, otherwise by val F1
        def get_f1_score(run):
            _, result = run
            return result.get('test_f1') or result.get('best_val_f1') or 0
            
        ranked_runs = sorted(successful_runs, key=get_f1_score, reverse=True)
        
        for i, (arch_name, result) in enumerate(ranked_runs, 1):
            f1_score = result.get('test_f1') or result.get('best_val_f1') or 0
            params = result.get('trainable_params', 'N/A')
            print(f"{i}. {arch_name.upper()}: F1={f1_score:.4f}, Params={params}")
            
    print(f"\n📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Successful: {len(successful_runs)}/{len(architectures)}")
    print(f"❌ Failed: {len(failed_runs)}/{len(architectures)}")
    
    # Save detailed results
    results_file = f"architecture_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"📁 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main() 