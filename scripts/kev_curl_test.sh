#!/usr/bin/env bash
# Kev (kev_pointer_v1) smoke test for a running vllm-jev server.
#
#   bash scripts/kev_curl_test.sh                  # default http://0.0.0.0:8001
#   BASE=http://<host>:8001 bash scripts/kev_curl_test.sh
#   CLAUSES=300 bash scripts/kev_curl_test.sh      # longer state (about 5k tokens)
#
# Never starts or stops a server. Needs curl and python3 (for JSON checks).
set -uo pipefail

BASE="${BASE:-http://0.0.0.0:8001}"
CLAUSES="${CLAUSES:-180}"   # about 17 tokens per clause
SALT="kev-curl-$(date +%s)"  # fresh per run, reused inside the run
# Never send localhost traffic through a corporate proxy.
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="$no_proxy"

pass=0
fail=0
ok()   { echo "  PASS  $*"; pass=$((pass + 1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail + 1)); }
step() { echo; echo "== $*"; }

STATE=$(python3 -c "import sys
print(' '.join(f'Clause {i}: the supplier shall deliver item {i} within 30 days.' for i in range(int(sys.argv[1]))))" "$CLAUSES")

# Write a request body with the shared state and the given questions JSON.
body() {
  python3 -c 'import json, sys
print(json.dumps({"state": sys.argv[1], "questions": json.loads(sys.argv[2]), "cache_salt": sys.argv[3]}))' \
    "$STATE" "$1" "$SALT"
}

# POST a body; prints the HTTP code on line 1 and the response on line 2.
post() {
  curl -sS -m 300 -o /tmp/kev_resp.json -w '%{http_code} %{time_total}\n' \
    -H 'Content-Type: application/json' -d "$1" "$BASE/v1/systemone"
}

metric() {
  curl -sS -m 30 "$BASE/metrics" | awk -v n="$1" '
    index($0, n) == 1 && substr($0, length(n) + 1, 1) ~ /[{ ]/ { s += $NF; f = 1 }
    END { if (f) printf "%.0f", s; else printf "absent" }'
}

echo "BASE=$BASE  state≈$((CLAUSES * 17)) tokens  salt=$SALT"

# 1. Liveness ----------------------------------------------------------------
step "1. health and model"
code=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$BASE/health")
[ "$code" = 200 ] && ok "/health 200" || { bad "/health $code (server down, or bound to 127.0.0.1?)"; exit 1; }
model=$(curl -sS -m 10 "$BASE/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
[ -n "$model" ] && ok "model: $model" || bad "/v1/models has no model id"

# 2. Three question types ----------------------------------------------------
step "2. noul / choice / score"
QUESTIONS='{
  "binding":    {"type": "noul",   "instructions": "Is delivery mandatory?"},
  "party":      {"type": "choice", "instructions": "Which party must deliver?",
                 "criteria": {"supplier": "The seller of the goods", "buyer": "The purchaser"}},
  "strictness": {"type": "score",  "instructions": "How strict is the deadline?",
                 "criteria": ["loose", "normal", "strict"]}
}'
read -r code secs <<<"$(post "$(body "$QUESTIONS")")"
if [ "$code" != 200 ]; then
  bad "HTTP $code: $(head -c 300 /tmp/kev_resp.json)"
else
  ok "HTTP 200 in ${secs}s"
  python3 - <<'PY' && ok "answers are well formed" || bad "answers are malformed"
import json, math
r = json.load(open("/tmp/kev_resp.json"))
a = r["answers"]
assert set(a) == {"binding", "party", "strictness"}, a.keys()
n = a["binding"]["noul"]
assert 0 <= n <= 1 and math.isfinite(n)
for k in ("party", "strictness"):
    p = a[k]["probabilities"]
    assert abs(sum(p.values()) - 1) < 1e-3, (k, sum(p.values()))
assert a["party"]["choice"] in ("supplier", "buyer")
print(f"    noul binding   = {n:.4f}")
print(f"    choice party   = {a['party']['choice']}  {a['party']['probabilities']}")
print(f"    score strict   = {a['strictness']['score']:.3f}  {a['strictness']['probabilities']}")
m = r.get("metadata", {})
print(f"    metadata       = cached_tokens={m.get('cached_tokens')} "
      f"uncached_reruns={m.get('uncached_reruns')} seconds={m.get('inference_seconds')}")
PY
fi

# 3. Invalid input must be rejected, not answered -----------------------------
step "3. invalid input returns 422"
code=$(curl -sS -m 30 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' \
  -d '{"state": "x", "questions": {"q": {"type": "nonsense"}}}' "$BASE/v1/systemone")
[ "$code" = 422 ] && ok "unknown question type -> 422" || bad "unknown question type -> $code"

# 4. Prefix cache --------------------------------------------------------------
# Repeat the same state under the same cache_salt. With the patch and
# VLLM_JEV_KEV_PREFIX_CACHE unset (on), hits must grow and cached_tokens > 0.
step "4. prefix cache reuse (same state, same cache_salt)"
q0=$(metric "vllm:prefix_cache_queries_total")
h0=$(metric "vllm:prefix_cache_hits_total")
cached=0
for i in 1 2 3; do
  read -r code secs <<<"$(post "$(body "$QUESTIONS")")"
  c=$(python3 -c 'import json; print(json.load(open("/tmp/kev_resp.json")).get("metadata",{}).get("cached_tokens",0))' 2>/dev/null || echo 0)
  echo "    round $i: HTTP $code  ${secs}s  cached_tokens=$c"
  cached=$((cached + c))
done
q1=$(metric "vllm:prefix_cache_queries_total")
h1=$(metric "vllm:prefix_cache_hits_total")
echo "    queries_total: $q0 -> $q1    hits_total: $h0 -> $h1"
if [ "$h1" = absent ]; then
  bad "server exposes no vllm:prefix_cache_hits_total"
elif [ "$((h1 - h0))" -gt 0 ] && [ "$cached" -gt 0 ]; then
  ok "cache hit: hits +$((h1 - h0)), cached_tokens total $cached"
elif [ "$((q1 - q0))" -eq 0 ]; then
  bad "no cache queries: the patch is not loaded, or VLLM_JEV_KEV_PREFIX_CACHE=0"
else
  bad "queried but never hit: check --enable-prefix-caching and that the state is long enough"
fi

# 5. Answers stay stable across cache hits -------------------------------------
step "5. repeated answer stability"
first=""
for i in 1 2; do
  post "$(body "$QUESTIONS")" >/dev/null
  pick=$(python3 -c 'import json; a=json.load(open("/tmp/kev_resp.json"))["answers"]; print(a["party"]["choice"], round(a["binding"]["noul"], 4))')
  echo "    round $i: $pick"
  [ -z "$first" ] && first="$pick"
done
[ "$pick" = "$first" ] && ok "same choice and noul (4 d.p.) on repeat" \
  || echo "  NOTE  answers moved between rounds; run scripts/kev_cache_ab.py to measure drift"

echo
echo "== result: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
