#!/usr/bin/env bash
set -euo pipefail

BIN_DIR="${HOME}/quantsociety/bin"
BUCKET="quantsociety-cold-data-1425188104"
USER_NAME="$(id -un)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cos_api_smoke_${USER_NAME}.XXXXXX")"

cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

need_cmd() {
  local cmd="$1"
  if [[ ! -x "${BIN_DIR}/${cmd}" ]]; then
    echo "MISSING ${BIN_DIR}/${cmd}" >&2
    exit 2
  fi
}

need_host_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "MISSING ${cmd}" >&2
    exit 2
  fi
}

require_403_with_request_id() {
  local label="$1" file="$2"
  if ! grep -q "HTTP 403" "$file"; then
    echo "${label}_UNEXPECTED_STATUS" >&2
    cat "$file" >&2
    exit 4
  fi
  if ! grep -q "request_id=" "$file"; then
    echo "${label}_MISSING_REQUEST_ID" >&2
    cat "$file" >&2
    exit 4
  fi
}

check_limits() {
  local file="$1"
  python3 - "$file" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)

required = {
    "ok",
    "transfer_mode",
    "max_actor_concurrent",
    "max_global_concurrent",
    "max_upload_bytes",
    "coscli_timeout_seconds",
}
missing = sorted(required - set(data))
if missing:
    raise SystemExit("limits missing keys: " + ",".join(missing))
if data.get("ok") is not True:
    raise SystemExit("limits ok is not true")
print(
    "LIMITS "
    f"transfer_mode={data['transfer_mode']} "
    f"max_actor_concurrent={data['max_actor_concurrent']} "
    f"max_global_concurrent={data['max_global_concurrent']} "
    f"max_upload_bytes={data['max_upload_bytes']} "
    f"coscli_timeout_seconds={data['coscli_timeout_seconds']}"
)
PY
}

run_allow_list() {
  local cmd="$1" uri="$2"
  local out="${WORK_DIR}/${cmd}_allow.txt"
  need_cmd "$cmd"
  "${BIN_DIR}/${cmd}" ls "$uri" >"$out"
  echo "ALLOW ${cmd} ${uri} count=$(wc -l <"$out")"
}

run_deny_list() {
  local cmd="$1" uri="$2"
  local out="${WORK_DIR}/${cmd}_deny.out"
  local err="${WORK_DIR}/${cmd}_deny.err"
  need_cmd "$cmd"
  if "${BIN_DIR}/${cmd}" ls "$uri" >"$out" 2>"$err"; then
    echo "DENY_FAILED ${cmd} ${uri}" >&2
    exit 3
  fi
  require_403_with_request_id "DENY" "$err"
  echo "DENY ${cmd} ${uri} $(head -1 "$err")"
}

run_deny_upload() {
  local cmd="$1" uri="$2"
  local out="${WORK_DIR}/${cmd}_deny_upload.out"
  local err="${WORK_DIR}/${cmd}_deny_upload.err"
  local probe="${WORK_DIR}/cos_api_smoke_deny_${USER_NAME}_$$.txt"
  need_cmd "$cmd"
  printf 'deny\n' > "$probe"
  if "${BIN_DIR}/${cmd}" cp "$probe" "$uri" >"$out" 2>"$err"; then
    echo "DENY_UPLOAD_FAILED ${cmd} ${uri}" >&2
    exit 5
  fi
  require_403_with_request_id "DENY_UPLOAD" "$err"
  echo "DENY_UPLOAD ${cmd} ${uri} $(head -1 "$err")"
}

run_allow_upload_delete() {
  local cmd="$1" uri="$2"
  local upload_out="${WORK_DIR}/${cmd}_allow_upload.out"
  local delete_out="${WORK_DIR}/${cmd}_allow_delete.out"
  local probe="${WORK_DIR}/cos_api_smoke_allow_${USER_NAME}_$$.txt"
  need_cmd "$cmd"
  printf 'allow\n' > "$probe"
  "${BIN_DIR}/${cmd}" cp "$probe" "$uri" >"$upload_out"
  "${BIN_DIR}/${cmd}" rm "$uri" >"$delete_out"
  echo "ALLOW_UPLOAD_DELETE ${cmd} ${uri}"
}

need_host_cmd python3
need_cmd cos-api
"${BIN_DIR}/cos-api" whoami
limits_out="${WORK_DIR}/limits.json"
"${BIN_DIR}/cos-api" limits >"$limits_out"
check_limits "$limits_out"

case "$USER_NAME" in
  yluel)
    run_allow_list raw-cos "cos://${BUCKET}/raw_data/"
    run_allow_list options-cos "cos://${BUCKET}/options/"
    run_deny_list raw-cos "cos://${BUCKET}/factor_pool/"
    ;;
  xlubs|gluoad|hsunbj)
    run_allow_list clean-cos-ro "cos://${BUCKET}/clean_data/"
    run_allow_list factor-cos "cos://${BUCKET}/factor_pool/"
    run_allow_upload_delete candidate-cos "cos://${BUCKET}/candidate_pool/_smoke_allow_${USER_NAME}.txt"
    run_deny_upload factor-cos "cos://${BUCKET}/factor_pool/_smoke_deny_${USER_NAME}.txt"
    ;;
  ychenql)
    run_allow_list clean-cos-ro "cos://${BUCKET}/clean_data/"
    run_allow_list factor-cos-ro "cos://${BUCKET}/factor_pool/"
    run_deny_list factor-cos-ro "cos://${BUCKET}/candidate_pool/"
    run_deny_upload factor-cos-ro "cos://${BUCKET}/factor_pool/_smoke_deny_${USER_NAME}.txt"
    ;;
  jlinm)
    run_allow_list options-cos "cos://${BUCKET}/options/"
    run_allow_upload_delete options-cos "cos://${BUCKET}/options/_smoke_allow_${USER_NAME}.txt"
    run_deny_list options-cos "cos://${BUCKET}/raw_data/"
    ;;
  *)
    echo "No smoke profile for ${USER_NAME}" >&2
    exit 7
    ;;
esac

echo "SMOKE_OK ${USER_NAME}"
