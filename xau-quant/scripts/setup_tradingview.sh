#!/usr/bin/env bash
# One command setup: aurum + TradingView MCP + TradingView Desktop + Claude Code.
#
#   cd resq-deck/xau-quant && ./scripts/setup_tradingview.sh
#
# What it does, in order:
#   1. checks git, node 18+, python 3.10+ and the claude CLI
#   2. clones (or updates) tradesdontlie/tradingview-mcp and installs it
#   3. creates .venv here and installs aurum with its MCP, LLM and data extras
#   4. compiles the aurum Pine script on TradingView's server (no chart needed)
#   5. registers both MCP servers with Claude Code for this folder
#   6. starts the aurum dashboard at http://127.0.0.1:8765
#   7. restarts TradingView Desktop with the debug port (asks first)
#   8. opens Claude Code with the run playbook (.claude/commands/tradingview-run.md)
#
# Options:
#   --yes          don't ask before restarting TradingView
#   --remote       start `claude remote-control` instead, to drive it from the Claude app
#   --no-launch    skip restarting TradingView
#   --no-claude    set everything up but don't start Claude Code
#   --no-web       don't start the dashboard
# Env: TRADINGVIEW_MCP_DIR (default ~/tradingview-mcp), TV_PORT (default 9222)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TV_DIR="${TRADINGVIEW_MCP_DIR:-$HOME/tradingview-mcp}"
TV_PORT="${TV_PORT:-9222}"
YES=0; REMOTE=0; LAUNCH=1; CLAUDE=1; WEB=1
for a in "$@"; do
  case "$a" in
    --yes|-y) YES=1 ;;
    --remote) REMOTE=1 ;;
    --no-launch) LAUNCH=0 ;;
    --no-claude) CLAUDE=0 ;;
    --no-web) WEB=0 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1;33m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '    \033[31m✗ %s\033[0m\n' "$*"; exit 1; }

# 1 ── prerequisites ────────────────────────────────────────────────────────
say "Checking prerequisites"
command -v git >/dev/null || die "git not found"
command -v node >/dev/null || die "Node.js 18+ not found: https://nodejs.org"
NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
[ "$NODE_MAJOR" -ge 18 ] || die "Node.js 18+ needed, found $(node --version)"
ok "node $(node --version)"
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || die "Python 3.10+ not found"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python 3.10+ needed, found $("$PY" --version)"
ok "$("$PY" --version)"
HAVE_CLAUDE=1
command -v claude >/dev/null || { HAVE_CLAUDE=0; warn "claude CLI not found: npm install -g @anthropic-ai/claude-code (MCP registration and the run step will be skipped)"; }

# 2 ── TradingView MCP ──────────────────────────────────────────────────────
say "TradingView MCP in $TV_DIR"
if [ -d "$TV_DIR/.git" ]; then
  git -C "$TV_DIR" pull --ff-only -q && ok "updated"
else
  git clone -q https://github.com/tradesdontlie/tradingview-mcp.git "$TV_DIR" && ok "cloned"
fi
(cd "$TV_DIR" && npm install --no-audit --no-fund --silent) && ok "npm install"

# 3 ── aurum ────────────────────────────────────────────────────────────────
say "aurum Python environment in $ROOT/.venv"
[ -x "$ROOT/.venv/bin/python" ] || "$PY" -m venv "$ROOT/.venv"
VPY="$ROOT/.venv/bin/python"
"$VPY" -m pip install -q --upgrade pip
"$VPY" -m pip install -q -e "$ROOT[mcp,llm,data]"
ok "aurum $("$VPY" -c 'import aurum; print(aurum.__version__)') installed"

# 4 ── compile the Pine script on TradingView's server ─────────────────────
say "Compiling tradingview/aurum_gold.pine on TradingView's server"
if CHECK="$(node "$TV_DIR/src/cli/index.js" pine check --file "$ROOT/tradingview/aurum_gold.pine" 2>&1)"; then
  echo "$CHECK" | sed 's/^/    /' | head -40
  if echo "$CHECK" | grep -q '"compiled": *true'; then
    ok "compiles"
  else
    warn "compile reported errors above; the Claude run will fix them"
  fi
else
  warn "compile check could not reach TradingView; the Claude run will compile on the chart instead"
  echo "$CHECK" | head -8 | sed 's/^/      /'
fi

# 5 ── register MCP servers with Claude Code (local scope, this folder) ────
if [ "$HAVE_CLAUDE" = 1 ]; then
  say "Registering MCP servers with Claude Code"
  cd "$ROOT"
  for name in aurum tradingview; do claude mcp remove -s local "$name" >/dev/null 2>&1 || true; done
  claude mcp add -s local aurum -- "$VPY" -m aurum.mcp_server >/dev/null && ok "aurum"
  claude mcp add -s local tradingview -- node "$TV_DIR/src/server.js" >/dev/null && ok "tradingview"
fi

# 6 ── dashboard ────────────────────────────────────────────────────────────
if [ "$WEB" = 1 ]; then
  say "Starting the aurum dashboard"
  mkdir -p "$HOME/.aurum"
  if curl -s -o /dev/null "http://127.0.0.1:8765/api/strategies"; then
    ok "already running at http://127.0.0.1:8765"
  else
    nohup "$VPY" -m aurum.web --port 8765 >"$HOME/.aurum/web.log" 2>&1 &
    sleep 2
    ok "http://127.0.0.1:8765  (log: ~/.aurum/web.log, stop: kill $!)"
  fi
  (command -v open >/dev/null && open "http://127.0.0.1:8765") || (command -v xdg-open >/dev/null && xdg-open "http://127.0.0.1:8765" >/dev/null 2>&1) || true
fi

# 7 ── TradingView Desktop with the debug port ─────────────────────────────
if [ "$LAUNCH" = 1 ]; then
  say "TradingView Desktop with --remote-debugging-port=$TV_PORT"
  if curl -s "http://127.0.0.1:$TV_PORT/json/version" >/dev/null 2>&1; then
    ok "already listening on $TV_PORT"
  else
    if [ "$YES" = 0 ]; then
      read -r -p "    This quits TradingView and reopens it in debug mode. Continue? [Y/n] " ans
      case "${ans:-Y}" in [Yy]*) ;; *) warn "skipped; start it yourself with --remote-debugging-port=$TV_PORT"; LAUNCH=0 ;; esac
    fi
    if [ "$LAUNCH" = 1 ]; then
      case "$(uname -s)" in
        Darwin) bash "$TV_DIR/scripts/launch_tv_debug_mac.sh" "$TV_PORT" || warn "launch failed (is TradingView Desktop installed?)" ;;
        Linux)  bash "$TV_DIR/scripts/launch_tv_debug_linux.sh" "$TV_PORT" || warn "launch failed (is TradingView Desktop installed?)" ;;
        *) warn "unknown OS; on Windows use scripts/setup_tradingview.ps1" ;;
      esac
    fi
  fi
fi

# 8 ── hand over to Claude Code ─────────────────────────────────────────────
PLAYBOOK="$ROOT/.claude/commands/tradingview-run.md"
if [ "$CLAUDE" = 1 ] && [ "$HAVE_CLAUDE" = 1 ]; then
  cd "$ROOT"
  if [ "$REMOTE" = 1 ]; then
    say "Starting Claude Code Remote Control. In the Claude app, open this session and send: /tradingview-run"
    exec claude remote-control
  fi
  say "Opening Claude Code with the run playbook (approve the two MCP servers if asked)"
  exec claude "$(cat "$PLAYBOOK")"
fi
say "Done. Run \`claude\` in $ROOT and type /tradingview-run"
