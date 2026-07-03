#!/usr/bin/env bash
# Страж ченжлога (git pre-push, НЕблокирующий): при пуше фоном выжимает из диффов
# юзерские фичи/фиксы через Claude Sonnet (headless `claude -p`) и шлёт готовые записи
# в бэкенд (POST /admin/changelog, идемпотентно по git-range). Пуш НЕ ждёт и НЕ ломается:
# рабочая часть уходит в setsid/nohup, любые ошибки — только в лог.
#
# Установка (в каждой репе): ln -sf ../../scripts/push-changelog.sh .git/hooks/pre-push
# Токен админа: ~/.config/norsk/changelog_token (Bearer). Логи: ~/.cache/norsk-changelog/<repo>.log
set -u

API="${NORSK_API:-https://api.learnnorsk.space}"
TOKEN_FILE="$HOME/.config/norsk/changelog_token"
LOG_DIR="$HOME/.cache/norsk-changelog"
DIFF_CAP=60000   # символов диффа в промпт (хватает на типичный пуш, не раздувает контекст)

repo_name() {
    case "$(basename "$(git rev-parse --show-toplevel)")" in
        learnNorskApp) echo frontend ;;
        laereNorskBackand) echo backend ;;
        *) basename "$(git rev-parse --show-toplevel)" ;;
    esac
}

# ── Рабочая часть (фон): собрать контекст → Sonnet → POST ────────────────────────────────
if [ "${1:-}" = "--worker" ]; then
    RANGE="$2"
    REPO="$(repo_name)"
    [ -f "$TOKEN_FILE" ] || { echo "[$(date -Is)] нет токена $TOKEN_FILE — пропуск"; exit 0; }
    LOG_TEXT="$(git log --reverse --format='--- %h %s%n%b' "$RANGE" 2>/dev/null)"
    [ -n "$LOG_TEXT" ] || { echo "[$(date -Is)] пустой range $RANGE — пропуск"; exit 0; }
    STAT="$(git diff --stat "$RANGE" 2>/dev/null | tail -40)"
    DIFF="$(git diff -U2 "$RANGE" -- . ':(exclude)*.lock' ':(exclude)*lock.json' 2>/dev/null | head -c "$DIFF_CAP")"
    # уже опубликованное (обе репы!) — чтобы фича, задетая и бэком, и фронтом, не анонсировалась дважды
    RECENT="$(curl -sS -m 15 "$API/changelog?limit=12" -H "Authorization: Bearer $(cat "$TOKEN_FILE")" \
        | python3 -c "import json,sys
try:
    for e in json.load(sys.stdin).get('entries', []):
        ru = e.get('i18n', {}).get('ru', {})
        print('- [' + e.get('kind','') + '] ' + (ru.get('t') or '') + ': ' + (ru.get('d') or ''))
except Exception:
    pass" 2>/dev/null)"

    PROMPT=$(cat <<EOF
Ты — редактор ченжлога приложения для изучения норвежского языка (веб-приложение: слова, SRS-повторения, игры, формы слов). Ниже коммиты и диффы одного пуша (репозиторий: $REPO).

Выпиши ТОЛЬКО изменения, заметные ПОЛЬЗОВАТЕЛЮ приложения (новые возможности, исправленные баги, ускорения, изменения интерфейса). Внутренние рефакторинги, тесты, CI, докстринги, дев-скрипты — пропусти. Если юзерских изменений нет — верни [].

Ответ — СТРОГО JSON-массив без пояснений и без markdown:
[{"kind":"feature|fix|perf|ui","i18n":{"ru":{"t":"...","d":"..."},"en":{"t":"...","d":"..."},"ukr":{"t":"...","d":"..."},"pl":{"t":"...","d":"..."},"lt":{"t":"...","d":"..."},"lv":{"t":"...","d":"..."},"ar":{"t":"...","d":"..."}}}]

Правила: t — до 55 знаков, d — до 160 знаков; живым человеческим языком, без жаргона разработки (никаких «рефакторинг», «эндпоинт», имён файлов/функций); пиши о пользе для учащегося. 1-4 записи максимум, объединяй мелочи.

ВАЖНО: ниже список УЖЕ ОПУБЛИКОВАННЫХ записей. НЕ создавай запись о том же самом (та же фича с другой стороны фронт/бэк, та же правка) — верни для неё ничего. Если ВСЁ уже покрыто — верни [].

=== уже опубликовано ===
$RECENT

=== git log ===
$LOG_TEXT

=== diffstat ===
$STAT

=== diff (обрезан) ===
$DIFF
EOF
)
    echo "[$(date -Is)] range=$RANGE repo=$REPO — спрашиваю sonnet…"
    CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || echo "$HOME/.local/bin/claude")}"
    RAW="$("$CLAUDE_BIN" -p --model sonnet "$PROMPT" 2>>"$LOG_DIR/$REPO.err")" || { echo "claude fail"; exit 0; }
    BODY="$(RAW="$RAW" REPO="$REPO" RANGE="$RANGE" python3 - <<'PY'
import json, os
raw = os.environ.get("RAW", "")
i, j = raw.find('['), raw.rfind(']')
try:
    entries = json.loads(raw[i:j + 1]) if i >= 0 and j > i else []
except Exception:
    entries = []
print(json.dumps({"repo": os.environ["REPO"], "source": os.environ["RANGE"],
                  "entries": entries}, ensure_ascii=False))
PY
)"
    N="$(python3 -c "import json,sys; print(len(json.loads(sys.argv[1])['entries']))" "$BODY" 2>/dev/null || echo 0)"
    if [ "$N" = "0" ]; then echo "[$(date -Is)] юзерских изменений нет — не шлю"; exit 0; fi
    RESP="$(curl -sS -m 30 -X POST "$API/admin/changelog" \
        -H "Authorization: Bearer $(cat "$TOKEN_FILE")" -H "Content-Type: application/json" \
        -d "$BODY")"
    echo "[$(date -Is)] отправлено $N записей → $RESP"
    exit 0
fi

# ── Хук-часть (мгновенная): вычислить range пуша и уйти в фон ────────────────────────────
RANGE=""
while read -r _local_ref local_sha _remote_ref remote_sha; do
    case "$local_sha" in *[!0]*) ;; *) continue ;; esac      # удаление ветки — мимо
    case "$remote_sha" in
        *[!0]*) RANGE="${remote_sha:0:12}..${local_sha:0:12}" ;;
        *) RANGE="" ;;                                        # новая ветка — истории «до» нет, пропуск
    esac
done
[ -n "$RANGE" ] || exit 0
mkdir -p "$LOG_DIR"
REPO="$(repo_name)"
setsid nohup "$0" --worker "$RANGE" >>"$LOG_DIR/$REPO.log" 2>&1 </dev/null &
exit 0
