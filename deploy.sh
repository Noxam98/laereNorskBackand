#!/usr/bin/env bash
# Деплой бэкенда на Fly.io. Запускать ПОСЛЕ `flyctl auth login`.
# Секреты берутся из .env (в репозиторий не коммитится).
set -euo pipefail

FLY="${FLY:-$HOME/.fly/bin/flyctl}"
APP="$(grep -E '^app\s*=' fly.toml | head -1 | sed -E 's/.*"(.*)".*/\1/')"
REGION="$(grep -E '^primary_region' fly.toml | head -1 | sed -E 's/.*"(.*)".*/\1/')"

echo "App: $APP  Region: $REGION"

# 1. Создать приложение (если ещё нет)
"$FLY" apps create "$APP" 2>/dev/null || echo "app exists, continue"

# 2. Том под SQLite (1GB), если ещё нет
if ! "$FLY" volumes list -a "$APP" 2>/dev/null | grep -q "data"; then
  "$FLY" volumes create data --region "$REGION" --size 1 -a "$APP" --yes
fi

# 3. Секреты из .env
set -a; source .env; set +a
"$FLY" secrets set -a "$APP" \
  LLM_BASE_URL="$LLM_BASE_URL" LLM_API_KEY="$LLM_API_KEY" LLM_MODEL="$LLM_MODEL" \
  EMBED_BASE_URL="$EMBED_BASE_URL" EMBED_API_KEY="$EMBED_API_KEY" EMBED_MODEL="$EMBED_MODEL" \
  SECRET_KEY="$SECRET_KEY" CORS_ORIGINS="${CORS_ORIGINS:-*}" \
  TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}" TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}" \
  TELEGRAM_ENABLED="${TELEGRAM_ENABLED:-true}" NOTIFY_COOLDOWN_SEC="${NOTIFY_COOLDOWN_SEC:-900}" \
  TELEGRAM_ADMIN_IDS="${TELEGRAM_ADMIN_IDS:-}" \
  TELEGRAM_FEED="${TELEGRAM_FEED:-false}" TELEGRAM_FEED_CHAT_ID="${TELEGRAM_FEED_CHAT_ID:-}"

# 4. Деплой (билд в облаке Fly, локальный docker не нужен)
"$FLY" deploy -a "$APP" --remote-only

echo "Done: https://$APP.fly.dev"
