#!/usr/bin/env bash
# Apply slime docker/patch stack (Qwen3-30B-A3B deterministic) inside a running
# container, then install miniTransformer with --no-deps (no dependency changes).
#
# Usage (inside container, e.g. root@f75cad06ec94):
#   bash /root/slime/scripts/setup-container-deterministic-patches.sh
#
# Optional env:
#   SLIME_DIR=/root/slime
#   MINITE_DIR=/root/miniTransformer
#   PATCH_VERSION=latest          # or v0.5.15.post1
#   MEGATRON_DIR=/root/Megatron-LM
#   SGLANG_DIR=/sgl-workspace/sglang
#   SKIP_MINITE=1                 # only apply patches
#   SKIP_PATCHES=1                # only install miniTransformer

set -euo pipefail

: "${SLIME_DIR:=/root/slime}"
: "${MINITE_DIR:=/root/miniTransformer}"
: "${PATCH_VERSION:=latest}"
: "${MEGATRON_DIR:=/root/Megatron-LM}"
: "${SGLANG_DIR:=/sgl-workspace/sglang}"
: "${SKIP_MINITE:=0}"
: "${SKIP_PATCHES:=0}"

PATCH_DIR="${SLIME_DIR}/docker/patch/${PATCH_VERSION}"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

apply_git_patch() {
    local target_dir="$1" patch_file="$2"
    [[ -f "$patch_file" ]] || fail "missing patch: $patch_file"

    pushd "$target_dir" >/dev/null
    if git apply --check --reverse "$patch_file" 2>/dev/null; then
        log "  [skip] already applied: $(basename "$patch_file")"
    elif git apply --check "$patch_file" 2>/dev/null; then
        git apply "$patch_file"
        log "  [done] applied: $(basename "$patch_file")"
    elif git apply --check --3way "$patch_file" 2>/dev/null; then
        git apply --3way "$patch_file"
        if grep -R -n '^<<<<<<< ' . >/dev/null 2>&1; then
            fail "3-way merge conflict in $(basename "$patch_file") under $target_dir"
        fi
        log "  [done] applied (3way): $(basename "$patch_file")"
    else
        fail "cannot apply $(basename "$patch_file") under $target_dir (version mismatch?)"
    fi
    popd >/dev/null
}

detect_patch_version() {
    if [[ -d "${SLIME_DIR}/docker/patch/${PATCH_VERSION}" ]]; then
        return 0
    fi
    for candidate in v0.5.15.post1 latest; do
        if [[ -d "${SLIME_DIR}/docker/patch/${candidate}" ]]; then
            PATCH_VERSION="$candidate"
            PATCH_DIR="${SLIME_DIR}/docker/patch/${PATCH_VERSION}"
            warn "PATCH_VERSION fallback -> ${PATCH_VERSION}"
            return 0
        fi
    done
    fail "no docker/patch/* directory under ${SLIME_DIR}"
}

apply_patches() {
    detect_patch_version
    [[ -d "$PATCH_DIR" ]] || fail "patch dir not found: $PATCH_DIR"
    [[ -d "$MEGATRON_DIR/.git" ]] || fail "Megatron git repo not found: $MEGATRON_DIR"
    [[ -d "$SGLANG_DIR/.git" ]] || fail "SGLang git repo not found: $SGLANG_DIR"

    log "Slime branch: $(git -C "$SLIME_DIR" branch --show-current 2>/dev/null || echo unknown)"
    log "Patch dir: $PATCH_DIR"
    log "Megatron: $(git -C "$MEGATRON_DIR" log -1 --oneline)"
    log "SGLang:   $(git -C "$SGLANG_DIR" log -1 --oneline)"

    log "Applying Megatron patches ..."
    apply_git_patch "$MEGATRON_DIR" "${PATCH_DIR}/megatron.patch"
    if [[ -f "${PATCH_DIR}/megatron-sglang-aligned.patch" ]]; then
        apply_git_patch "$MEGATRON_DIR" "${PATCH_DIR}/megatron-sglang-aligned.patch"
    fi

    log "Applying SGLang patches (deterministic + Qwen3) ..."
    local sglang_patches=(
        sglang.patch
        sglang-top_p.patch
        sglang-release_hicache.patch
        sglang-pull_weights.patch
        sglang-deterministic.patch
        sglang-qwen3-deterministic.patch
    )
    for patch in "${sglang_patches[@]}"; do
        local patch_path="${PATCH_DIR}/${patch}"
        if [[ ! -f "$patch_path" ]]; then
            warn "  [skip] not in repo: $patch"
            continue
        fi
        apply_git_patch "$SGLANG_DIR" "$patch_path"
    done

    log "Patch apply complete."
}

install_minitransformer() {
    [[ -d "$MINITE_DIR" ]] || fail "miniTransformer not found: $MINITE_DIR"
    log "miniTransformer: $(git -C "$MINITE_DIR" log -1 --oneline 2>/dev/null || ls "$MINITE_DIR")"

    pushd "$MINITE_DIR" >/dev/null
    if [[ -f .gitmodules ]]; then
        log "Updating git submodules ..."
        git submodule update --init --recursive
    fi

    log "Building + installing miniTransformer (editable, --no-deps) ..."
    # --no-deps: do not touch torch/TE/other container deps
    # --no-build-isolation: compile fp4_gemm against the container's torch
    MAX_JOBS="${MAX_JOBS:-$(nproc)}"
    pip install -e . --no-deps --no-build-isolation

    python3 - <<'PY'
import importlib
importlib.import_module("fp4_gemm")
print("fp4_gemm import OK")
PY
    popd >/dev/null
    log "miniTransformer install complete."
}

main() {
    log "=== container deterministic patch + miniTE setup ==="
    [[ -d "$SLIME_DIR" ]] || fail "SLIME_DIR not found: $SLIME_DIR"

    if [[ "$SKIP_PATCHES" != "1" ]]; then
        apply_patches
    else
        log "SKIP_PATCHES=1 — skipping patch apply"
    fi

    if [[ "$SKIP_MINITE" != "1" ]]; then
        install_minitransformer
    else
        log "SKIP_MINITE=1 — skipping miniTransformer install"
    fi

    log "All done."
}

main "$@"
