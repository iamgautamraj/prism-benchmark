"""
RPAC Benchmark - Main Runner
Runs the benchmark across multiple LLMs and generates reports
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import csv

from prompts import PROMPTS as PROMPTS_V1
from prompts_v2 import PROMPTS_V2
from prompts_optimized import PROMPTS_OPTIMIZED
from llm_client import create_client, get_api_key, LLMClient
from judge import judge_response, JudgeResult


@dataclass
class BenchmarkResult:
    """Result for a single prompt-model evaluation."""
    prompt_id: str
    rpac_id: str
    domain: str
    model_name: str
    prompt_text: str
    model_response: str
    paas: float
    cns: float
    fni: float
    rpac_match: Optional[str]
    justification: str
    timestamp: str
    raw_judge_response: Optional[str] = None  # For debugging


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def evaluate_single_prompt(
    model_client: LLMClient,
    judge_client: LLMClient,
    model_name: str,
    prompt_data: dict,
    max_tokens: int,
    temperature: float,
    prompt_index: int,
    total_prompts: int
) -> BenchmarkResult:
    """Evaluate a single prompt - used for parallel processing."""
    try:
        # Get model response
        model_response = model_client.generate(
            prompt=prompt_data["prompt"],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Judge the response
        judge_result = judge_response(
            judge_client=judge_client,
            user_prompt=prompt_data["prompt"],
            model_response=model_response
        )
        
        result = BenchmarkResult(
            prompt_id=prompt_data["id"],
            rpac_id=prompt_data["rpac_id"],
            domain=prompt_data["domain"],
            model_name=model_name,
            prompt_text=prompt_data["prompt"],
            model_response=model_response,
            paas=judge_result.paas,
            cns=judge_result.cns,
            fni=judge_result.fni,
            rpac_match=judge_result.rpac_match,
            justification=judge_result.justification,
            timestamp=datetime.now().isoformat(),
            raw_judge_response=judge_result.raw_response
        )
        
        print(f"  ✓ [{prompt_index}/{total_prompts}] {prompt_data['id']}: PAAS={result.paas:.1f} CNS={result.cns:.1f} FNI={result.fni:.2f}")
        return result
        
    except Exception as e:
        print(f"  ✗ [{prompt_index}/{total_prompts}] {prompt_data['id']}: ERROR: {e}")
        return BenchmarkResult(
            prompt_id=prompt_data["id"],
            rpac_id=prompt_data["rpac_id"],
            domain=prompt_data["domain"],
            model_name=model_name,
            prompt_text=prompt_data["prompt"],
            model_response=f"ERROR: {e}",
            paas=0.0,
            cns=0.0,
            fni=1.0,
            rpac_match=None,
            justification=f"Error during evaluation: {e}",
            timestamp=datetime.now().isoformat()
        )


def run_benchmark(
    model_client: LLMClient,
    judge_client: LLMClient,
    model_name: str,
    prompts: List[dict],
    max_tokens: int = 2048,
    temperature: float = 0.7,
    max_workers: int = 3
) -> List[BenchmarkResult]:
    """
    Run benchmark for a single model with parallel processing.
    
    Args:
        model_client: LLM client for the model being evaluated
        judge_client: LLM client for the judge model
        model_name: Name of the model being evaluated
        prompts: List of prompt dictionaries
        max_tokens: Max tokens for model responses
        temperature: Temperature for model responses
        max_workers: Max parallel workers (default: 3 to respect rate limits)
    
    Returns:
        List of BenchmarkResult objects
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    total = len(prompts)
    results = []
    
    print(f"  Running with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_prompt = {
            executor.submit(
                evaluate_single_prompt,
                model_client, judge_client, model_name,
                prompt_data, max_tokens, temperature,
                i, total
            ): prompt_data
            for i, prompt_data in enumerate(prompts, 1)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_prompt):
            result = future.result()
            results.append(result)
    
    # Sort by prompt_id to maintain order
    results.sort(key=lambda r: r.prompt_id)
    return results


def compute_aggregate_scores(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute aggregate scores for a model's results."""
    if not results:
        return {}
    
    valid_results = [r for r in results if "ERROR" not in r.model_response]
    
    if not valid_results:
        return {"error": "All evaluations failed"}
    
    paas_scores = [r.paas for r in valid_results]
    cns_scores = [r.cns for r in valid_results]
    fni_scores = [r.fni for r in valid_results]
    
    return {
        "model": results[0].model_name,
        "total_prompts": len(results),
        "successful_evaluations": len(valid_results),
        "mean_paas": sum(paas_scores) / len(paas_scores),
        "mean_cns": sum(cns_scores) / len(cns_scores),
        "mean_fni": sum(fni_scores) / len(fni_scores),
        "false_novelty_rate": sum(1 for f in fni_scores if f >= 0.5) / len(fni_scores),
        "high_paas_rate": sum(1 for p in paas_scores if p >= 0.7) / len(paas_scores),
        "domain_breakdown": compute_domain_breakdown(valid_results)
    }


def compute_domain_breakdown(results: List[BenchmarkResult]) -> Dict[str, Dict[str, float]]:
    """Compute scores broken down by domain."""
    domains = {}
    for r in results:
        if r.domain not in domains:
            domains[r.domain] = {"paas": [], "cns": [], "fni": []}
        domains[r.domain]["paas"].append(r.paas)
        domains[r.domain]["cns"].append(r.cns)
        domains[r.domain]["fni"].append(r.fni)
    
    return {
        domain: {
            "mean_paas": sum(scores["paas"]) / len(scores["paas"]),
            "mean_cns": sum(scores["cns"]) / len(scores["cns"]),
            "mean_fni": sum(scores["fni"]) / len(scores["fni"]),
            "count": len(scores["paas"])
        }
        for domain, scores in domains.items()
    }


def save_results(
    results: List[BenchmarkResult],
    aggregate: Dict[str, Any],
    output_dir: str,
    model_name: str
):
    """Save benchmark results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    
    # Save detailed results as JSON
    json_path = output_path / f"{safe_model_name}_{timestamp}_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "aggregate": aggregate,
            "results": [asdict(r) for r in results]
        }, f, indent=2)
    print(f"  Saved JSON results to: {json_path}")
    
    # Save as CSV for easy analysis
    csv_path = output_path / f"{safe_model_name}_{timestamp}_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "prompt_id", "rpac_id", "domain", "model_name",
            "paas", "cns", "fni", "rpac_match", "justification"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "prompt_id": r.prompt_id,
                "rpac_id": r.rpac_id,
                "domain": r.domain,
                "model_name": r.model_name,
                "paas": r.paas,
                "cns": r.cns,
                "fni": r.fni,
                "rpac_match": r.rpac_match,
                "justification": r.justification
            })
    print(f"  Saved CSV results to: {csv_path}")
    
    # Save aggregate summary
    summary_path = output_path / f"{safe_model_name}_{timestamp}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"  Saved summary to: {summary_path}")


def print_summary(aggregate: Dict[str, Any]):
    """Print a formatted summary of results."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK SUMMARY: {aggregate.get('model', 'Unknown')}")
    print("=" * 60)
    print(f"  Prompts evaluated: {aggregate.get('successful_evaluations', 0)}/{aggregate.get('total_prompts', 0)}")
    print(f"  Mean PAAS (Prior-Art Awareness):  {aggregate.get('mean_paas', 0):.2f}")
    print(f"  Mean CNS (Conceptual Novelty):    {aggregate.get('mean_cns', 0):.2f}")
    print(f"  Mean FNI (False Novelty Index):   {aggregate.get('mean_fni', 0):.2f}")
    print(f"  False Novelty Rate (FNI ≥ 0.5):   {aggregate.get('false_novelty_rate', 0):.1%}")
    print(f"  High PAAS Rate (PAAS ≥ 0.7):      {aggregate.get('high_paas_rate', 0):.1%}")
    
    if "domain_breakdown" in aggregate:
        print("\n  Domain Breakdown:")
        for domain, scores in aggregate["domain_breakdown"].items():
            print(f"    {domain}: PAAS={scores['mean_paas']:.2f} CNS={scores['mean_cns']:.2f} FNI={scores['mean_fni']:.2f} (n={scores['count']})")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run RPAC Benchmark on LLMs")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--models", nargs="*", help="Specific models to test (by name)")
    parser.add_argument("--prompts", nargs="*", help="Specific prompt IDs to test (e.g., P01 P05)")
    parser.add_argument("--output", help="Override output directory")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers (default: 3)")
    parser.add_argument("--prompt-version", choices=["v1", "v2", "opt"], default="v1",
                        help="Prompt version: v1=generation (81), v2=validation (67), opt=optimized (30)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    output_dir = args.output or config.get("output_dir", "./results")
    
    # Select prompt version
    if args.prompt_version == "v2":
        PROMPTS = PROMPTS_V2
        print("Using PROMPTS_V2 (67 validation-style prompts)")
    elif args.prompt_version == "opt":
        PROMPTS = PROMPTS_OPTIMIZED
        print("Using PROMPTS_OPTIMIZED (30 cost-effective prompts)")
    else:
        PROMPTS = PROMPTS_V1
        print("Using PROMPTS_V1 (81 generation-style prompts)")
    
    # Filter prompts if specified
    prompts_to_run = PROMPTS
    if args.prompts:
        prompts_to_run = [p for p in PROMPTS if p["id"] in args.prompts]
        print(f"Running {len(prompts_to_run)} selected prompts")
    
    # Create judge client
    judge_config = config["judge_model"]
    judge_client = create_client(
        provider=judge_config["provider"],
        api_key=get_api_key(judge_config["api_key_env"]),
        model_id=judge_config["model_id"]
    )
    print(f"Judge model: {judge_config['model_id']}")
    
    # Filter models if specified
    models_to_test = config["models"]
    if args.models:
        models_to_test = [m for m in models_to_test if m["name"] in args.models]
    
    all_aggregates = []
    
    # Run benchmark for each model
    for model_config in models_to_test:
        model_name = model_config["name"]
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        try:
            model_client = create_client(
                provider=model_config["provider"],
                api_key=get_api_key(model_config["api_key_env"]),
                model_id=model_config["model_id"]
            )
            
            results = run_benchmark(
                model_client=model_client,
                judge_client=judge_client,
                model_name=model_name,
                prompts=prompts_to_run,
                max_tokens=config.get("max_tokens", 2048),
                temperature=config.get("temperature", 0.7),
                max_workers=args.workers
            )
            
            aggregate = compute_aggregate_scores(results)
            all_aggregates.append(aggregate)
            
            save_results(results, aggregate, output_dir, model_name)
            print_summary(aggregate)
            
        except Exception as e:
            print(f"ERROR evaluating {model_name}: {e}")
            continue
    
    # Save comparative summary
    if len(all_aggregates) > 1:
        comparative_path = Path(output_dir) / f"comparative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparative_path, "w") as f:
            json.dump(all_aggregates, f, indent=2)
        print(f"\nSaved comparative results to: {comparative_path}")


if __name__ == "__main__":
    main()
