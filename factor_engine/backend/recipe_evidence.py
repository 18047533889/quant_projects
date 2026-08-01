# -*- coding: utf-8 -*-
"""Fail-closed execution evidence for factor recipes.

A production recipe needs at least one real certified execution path after DAG
expansion.  The historical three-backend corpus remains a stronger portability
tier, not a prerequisite for every domain recipe.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

FE_ROOT=Path(__file__).resolve().parents[1]
CASE_PATH=FE_ROOT/"evidence"/"recipe_case_registry.json"
VERIFIED_PATH=FE_ROOT/"evidence"/"recipe_verified.json"

def _hash_file(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _hash_tree(root:Path)->str:
    digest=hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:continue
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()
def _production_recipe_names()->set[str]:
    from factor_recipes import FactorRecipeRegistry
    return set(FactorRecipeRegistry.list_names(status="production"))
def current_hashes()->dict[str,str]:
    return {
        "case_registry_hash":_hash_file(CASE_PATH),
        "recipe_source_hash":_hash_tree(FE_ROOT/"factor_recipes"),
        "portable_test_source_hash":_hash_file(FE_ROOT/"tests"/"operators"/"test_recipe_backend_certification.py"),
        "production_test_source_hash":_hash_file(FE_ROOT/"tests"/"operators"/"test_recipe_production_admission_v2.py"),
        "primitive_evidence_hash":_hash_file(FE_ROOT/"evidence"/"primitive_verified.json"),
        "factor_operator_evidence_hash":_hash_file(FE_ROOT/"evidence"/"factor_operator_verified.json"),
    }
def recipe_evidence_valid(*,require_commit_ancestor:bool=False)->bool:
    try:payload=json.loads(VERIFIED_PATH.read_text(encoding="utf-8"));case=json.loads(CASE_PATH.read_text(encoding="utf-8"))
    except (OSError,ValueError,TypeError):return False
    if payload.get("artifact_kind")!="test_passed" or payload.get("hashes")!=current_hashes():return False
    if set(payload.get("recipes") or [])!=_production_recipe_names():return False
    if set(payload.get("portable_three_backend_recipes") or [])!=set(case.get("recipes") or []):return False
    if not set(payload.get("recipes") or []).issubset(set((payload.get("certified_backends") or {}).keys())):return False
    if any(not (payload.get("certified_backends") or {}).get(name) for name in payload.get("recipes") or []):return False
    commit=str(payload.get("commit_sha") or "")
    if require_commit_ancestor:
        if not commit or subprocess.run(["git","merge-base","--is-ancestor",commit,"HEAD"],cwd=FE_ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False).returncode!=0:return False
    return True
def recipe_execution_verified(name:str)->bool:
    if not recipe_evidence_valid():return False
    payload=json.loads(VERIFIED_PATH.read_text(encoding="utf-8"));return str(name) in set(payload.get("recipes") or [])
def recipe_certified_backends(name:str)->frozenset[str]:
    if not recipe_evidence_valid():return frozenset()
    payload=json.loads(VERIFIED_PATH.read_text(encoding="utf-8"));return frozenset(str(x) for x in (payload.get("certified_backends") or {}).get(str(name),[]))
def verified_recipe_names()->frozenset[str]:
    if not recipe_evidence_valid():return frozenset()
    payload=json.loads(VERIFIED_PATH.read_text(encoding="utf-8"));return frozenset(str(x) for x in payload.get("recipes") or [])
